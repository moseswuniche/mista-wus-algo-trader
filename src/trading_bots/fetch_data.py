import os
import pandas as pd
from datetime import datetime
import time
import argparse
from typing import List, Optional, Dict, Tuple
import logging
from multiprocessing import Pool, cpu_count

from binance.client import Client
from binance.exceptions import BinanceAPIException

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "XRPUSDT",
    "ALGOUSDT",
    "ETHUSDT",
    "HBARUSDT",
    "ADAUSDT",
    "SOLUSDT",
    "LTCUSDT",
    "SUIUSDT",
]
DEFAULT_INTERVAL = Client.KLINE_INTERVAL_1DAY  # Daily data
DEFAULT_START_DATE = "2017-01-01"
DEFAULT_OUTPUT_DIR = "data"
BINANCE_API_LIMIT = 1000  # Default limit per request
DEFAULT_FETCH_PROCESSES = max(1, cpu_count() // 2)  # Use half the cores by default


def fetch_historical_klines(
    client: Client,
    symbol: str,
    interval: str,
    start_str: str,
    end_str: Optional[str] = None,
) -> List:
    """Fetches historical klines in batches, handling API limits."""
    logger.info(
        f"Fetching {symbol} {interval} klines from {start_str} to {end_str or 'now'}..."
    )
    all_klines = []
    start_dt_ms = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_dt_ms = (
        int(datetime.now().timestamp() * 1000)
        if end_str is None
        else int(datetime.strptime(end_str, "%Y-%m-%d").timestamp() * 1000)
    )
    limit = BINANCE_API_LIMIT
    total_fetched_count = 0

    while True:
        try:
            logger.debug(
                f"Fetching chunk starting from {datetime.fromtimestamp(start_dt_ms / 1000)}..."
            )
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_dt_ms,
                end_str=end_dt_ms,
                limit=limit,
            )

            if not klines:
                logger.info("No more data received for this period.")
                break

            chunk_size = len(klines)
            total_fetched_count += chunk_size
            all_klines.extend(klines)

            last_kline_close_time = klines[-1][6]
            start_dt_ms = last_kline_close_time + 1

            logger.info(
                f"Fetched chunk of {chunk_size} klines. Last timestamp: {datetime.fromtimestamp(last_kline_close_time / 1000)}. Total fetched for {symbol}: {total_fetched_count}"
            )

            # Respect API limits - a small delay between requests
            time.sleep(0.2)

            # Check if we have fetched up to the end date (if specified) or beyond current time
            if start_dt_ms >= end_dt_ms:
                logger.info("Reached end date/time.")
                break

        except BinanceAPIException as bae:
            # Handle specific API errors (e.g., rate limits, invalid symbol)
            logger.error(
                f"Binance API Error for {symbol}: Code={bae.code}, Message={bae.message}"
            )
            if bae.code == -1121:  # Invalid symbol
                logger.error(f"Invalid symbol: {symbol}. Skipping...")
                return []  # Return empty list for invalid symbol
            elif bae.code == -1003:  # Rate limit
                logger.warning("Rate limit hit. Waiting longer...")
                time.sleep(60)  # Wait longer for rate limits
            else:
                time.sleep(5)  # General API error wait
            continue  # Retry after waiting

        except Exception as e:
            logger.error(
                f"An unexpected error occurred fetching data for {symbol}: {e}",
                exc_info=True,
            )
            time.sleep(5)  # Wait after unexpected error
            continue  # Retry

    logger.info(
        f"Finished fetching for {symbol}. Total klines received: {len(all_klines)}"
    )
    return all_klines


def process_klines_to_dataframe(klines: List) -> pd.DataFrame:
    """Converts raw kline data to a pandas DataFrame."""
    if not klines:
        logger.info("No klines to process into DataFrame.")
        return pd.DataFrame()

    logger.info(f"Processing {len(klines)} klines into DataFrame...")
    df = pd.DataFrame(
        klines,
        columns=[
            "Open Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close Time",
            "Quote Asset Volume",
            "Number of Trades",
            "Taker Buy Base Asset Volume",
            "Taker Buy Quote Asset Volume",
            "Ignore",
        ],
    )

    # Convert timestamps to datetime objects (using Open Time for the main Date index)
    df["Date"] = pd.to_datetime(df["Open Time"], unit="ms")
    df.set_index("Date", inplace=True)

    # Select and rename relevant columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Convert OHLCV columns to numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove duplicates just in case (based on index)
    df = df[~df.index.duplicated(keep="first")]

    # Sort by date
    df.sort_index(inplace=True)

    logger.info(f"Finished processing. DataFrame shape: {df.shape}")
    return df


# --- Worker Function for Parallel Fetching ---
def fetch_and_save_symbol(
    args_tuple: Tuple[str, str, str, Optional[str], str],
) -> Tuple[str, bool, Optional[str]]:
    """Worker function to fetch, process, and save data for a single symbol."""
    symbol, interval, start_str, end_str, output_dir = args_tuple
    worker_logger = logging.getLogger(
        f"fetch_worker_{symbol}"
    )  # Worker-specific logger
    worker_logger.info(f"--- Starting fetch for {symbol} --- ")

    try:
        # Initialize client within the worker
        client = Client()

        # 1. Fetch data
        klines = fetch_historical_klines(
            client=client,
            symbol=symbol,
            interval=interval,
            start_str=start_str,
            end_str=end_str,
        )

        if not klines:
            msg = f"No klines fetched for {symbol}. Skipping processing and saving."
            worker_logger.warning(msg)
            return symbol, False, msg

        # 2. Process data
        df = process_klines_to_dataframe(klines)

        # 3. Save data
        if not df.empty:
            interval_suffix = interval
            filename = f"{symbol}_{interval_suffix}.csv"
            # Ensure the interval directory exists within the main output directory
            interval_dir = os.path.join(output_dir, interval_suffix)
            os.makedirs(interval_dir, exist_ok=True)  # Create interval subdir
            filepath = os.path.join(interval_dir, filename)
            worker_logger.info(f"Saving data for {symbol} to {filepath}...")
            df.to_csv(filepath)
            success_msg = f"Data for {symbol} saved successfully to {filepath}"
            worker_logger.info(success_msg)
            return symbol, True, success_msg
        else:
            msg = f"No data processed for {symbol}. Skipping save."
            worker_logger.warning(msg)
            return symbol, False, msg

    except Exception as e:
        error_msg = f"!!! Failed to process {symbol}: {e} !!!"
        worker_logger.critical(error_msg, exc_info=True)
        return symbol, False, error_msg


# --- End Worker Function ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch historical Klines from Binance."
    )
    parser.add_argument(
        "-s",
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=f"List of symbols to fetch (default: {' '.join(DEFAULT_SYMBOLS)}).",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=str,
        default=DEFAULT_INTERVAL,
        help=f"Kline interval (e.g., 1m, 5m, 1h, 1d, default: {DEFAULT_INTERVAL}).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_START_DATE,
        help=f"Start date in YYYY-MM-DD format (default: {DEFAULT_START_DATE}).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (default: current date).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save CSV files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. Default: INFO",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=DEFAULT_FETCH_PROCESSES,
        help=f"Number of parallel processes to use for fetching (default: {DEFAULT_FETCH_PROCESSES}).",
    )

    args = parser.parse_args()

    # Set logging level from args
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    # Configure root logger for main process
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)  # Main logger

    # Create base output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Binance client (no API key needed for public data) - MOVED TO WORKER
    # client = Client()

    logger.info(f"Starting parallel data fetch with args: {args}")

    # --- Prepare arguments for workers ---
    tasks = []
    for symbol in args.symbols:
        tasks.append((symbol, args.interval, args.start, args.end, args.output_dir))

    # --- Run tasks in parallel ---
    num_processes = min(
        args.processes, len(tasks)
    )  # Don't use more processes than tasks
    logger.info(
        f"Fetching data for {len(tasks)} symbols using {num_processes} processes..."
    )

    successful_fetches = 0
    failed_fetches = 0

    try:
        with Pool(processes=num_processes) as pool:
            results = pool.map(fetch_and_save_symbol, tasks)

        logger.info("--- Processing fetch results ---")
        for symbol, success, message in results:
            if success:
                successful_fetches += 1
                # Main logger can still log success if needed, or rely on worker logs
                # logger.info(message)
            else:
                failed_fetches += 1
                logger.error(f"Fetch failed for {symbol}: {message}")

    except Exception as e:
        logger.critical(
            f"Error during parallel fetching pool execution: {e}", exc_info=True
        )
        failed_fetches = len(tasks)  # Assume all failed if pool crashed

    # --- Summary ---
    logger.info("--- Data Fetching Process Summary ---")
    logger.info(f"  Total symbols attempted: {len(tasks)}")
    logger.info(f"  Successful fetches:      {successful_fetches}")
    logger.info(f"  Failed fetches:          {failed_fetches}")
    logger.info("--- Data fetching process finished. ---")

    # --- Old Sequential Logic (Removed) ---
    # for symbol in args.symbols:
    #     logger.info(f"--- Processing {symbol} --- ")
    #     try:
    #         # Fetch data
    #         klines = fetch_historical_klines(
    #             client=client,
    #             symbol=symbol,
    #             interval=args.interval,
    #             start_str=args.start,
    #             end_str=args.end,
    #         )
    #
    #         if not klines:
    #             logger.warning(
    #                 f"No klines fetched for {symbol}. Skipping processing and saving."
    #             )
    #             continue
    #
    #         # Process data
    #         df = process_klines_to_dataframe(klines)
    #
    #         # Save data
    #         if not df.empty:
    #             interval_suffix = args.interval
    #             filename = f"{symbol}_{interval_suffix}.csv"
    #             filepath = os.path.join(args.output_dir, filename)
    #             logger.info(f"Saving data for {symbol} to {filepath}...")
    #             df.to_csv(filepath)
    #             logger.info(f"Data for {symbol} saved successfully to {filepath}")
    #         else:
    #             logger.warning(f"No data processed for {symbol}. Skipping save.")
    #
    #     except Exception as e:
    #         logger.critical(f"!!! Failed to process {symbol}: {e} !!!", exc_info=True)
    #         continue
    # logger.info("--- Data fetching process finished. ---")
