#!/usr/bin/env python
import argparse
import subprocess
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
import shlex
from typing import List, Tuple, Optional

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---
# DEFAULT_PROCESSES = max(1, cpu_count() - 1)  # Old default
DEFAULT_PROCESSES = 6  # Optimized default for M3 Pro (typically 6 Performance cores)


def run_single_optimization(args_tuple: Tuple) -> Tuple[str, int, str, str]:
    """
    Worker function to run a single optimization command using subprocess.

    Args:
        args_tuple: A tuple containing (strategy, symbol, file_path_str,
                     common_args_str, job_index, total_jobs).

    Returns:
        A tuple containing (job_description, return_code, stdout, stderr).
    """
    (
        strategy,
        symbol,
        file_path_str,
        common_args_str,
        job_index,
        total_jobs,
    ) = args_tuple

    job_description = f"Optimize {strategy} {symbol} ({job_index}/{total_jobs})"
    logger.info(f"--- Starting: {job_description} ---")

    # Construct the specific arguments for this run
    specific_args = (
        f"--strategy {strategy} --symbol {symbol} --file {shlex.quote(file_path_str)}"
    )

    # Combine specific and common arguments
    full_args_str = f"{specific_args} {common_args_str}"

    # Construct the full command using poetry run
    # Ensure the command is correctly tokenized for subprocess
    command = [
        "poetry",
        "run",
        "python",
        "-m",
        "src.trading_bots.optimize",
    ] + shlex.split(
        full_args_str
    )  # Split the args string safely

    logger.debug(
        f"Executing command: {' '.join(command)}"
    )  # Log the command for debugging

    try:
        # Execute the command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
        )

        stdout_log = result.stdout.strip()
        stderr_log = result.stderr.strip()

        if result.returncode == 0:
            logger.info(f"--- Finished successfully: {job_description} ---")
            if stdout_log:
                logger.info(f"Stdout: \n{stdout_log}")  # Log stdout only if not empty
            if stderr_log:
                logger.warning(
                    f"Stderr: \n{stderr_log}"
                )  # Log stderr as warning if not empty
        else:
            logger.error(
                f"--- Failed: {job_description} (Exit Code: {result.returncode}) ---"
            )
            if stdout_log:
                logger.error(f"Stdout: \n{stdout_log}")
            if stderr_log:
                logger.error(f"Stderr: \n{stderr_log}")

        return (job_description, result.returncode, stdout_log, stderr_log)

    except Exception as e:
        logger.error(
            f"--- Exception during: {job_description} - {e} ---", exc_info=True
        )
        return (job_description, -1, "", str(e))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch strategy optimization in parallel."
    )

    # --- Core Arguments ---
    parser.add_argument(
        "--strategies",
        type=str,
        required=True,
        help="Space-separated list of strategy names.",
    )
    parser.add_argument(
        "--symbols", type=str, required=True, help="Space-separated list of symbols."
    )
    parser.add_argument(
        "--interval", type=str, required=True, help="Data interval (e.g., 1h)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Base directory for CSV data files.",
    )

    # --- Optimization Parameters ---
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Optimization start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Optimization end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Commission in basis points (e.g., 7.5 for 0.075%).",
    )
    parser.add_argument(
        "--balance", type=float, default=10000.0, help="Initial balance for backtests."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cumulative_profit",
        help="Metric for optimization.",
    )
    parser.add_argument(
        "--config",  # Added config path
        default="config/optimize_params.yaml",
        help="Path to the optimization parameters YAML file.",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",  # Handled by $(if ...) in Makefile
        help="Save detailed optimization results.",
    )
    parser.add_argument(
        "--details-file",
        type=str,
        default=None,
        help="Optional *base* path for detailed results CSV (will be suffixed).",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=DEFAULT_PROCESSES,
        help=f"Number of parallel processes to use (default: {DEFAULT_PROCESSES}).",
    )

    # --- Filter Arguments (Passed from Makefile) ---
    parser.add_argument(
        "--apply-atr-filter",
        action="store_true",  # Handled by $(if ...) in Makefile
        help="Apply ATR volatility filter.",
    )
    parser.add_argument(
        "--atr-filter-period",
        type=int,
        default=14,
        help="Period for ATR calculation.",
    )
    parser.add_argument(
        "--atr-filter-multiplier",
        type=float,
        default=1.5,
        help="Multiplier for ATR volatility threshold.",
    )
    parser.add_argument(
        "--atr-filter-sma-period",
        type=int,
        default=100,
        help="SMA period for ATR threshold baseline.",
    )
    parser.add_argument(
        "--apply-seasonality-filter",
        action="store_true",  # Handled by $(if ...) in Makefile
        help="Apply seasonality filter (trading hours).",
    )
    parser.add_argument(
        "--allowed-trading-hours-utc",
        type=str,
        default="",
        help="Allowed trading hours in UTC (e.g., '5-17').",
    )
    parser.add_argument(
        "--apply-seasonality-to-symbols",
        type=str,
        default="",
        help="Comma-separated list of symbols to apply seasonality filter to.",
    )

    args = parser.parse_args()

    strategies = args.strategies.split()
    symbols = args.symbols.split()
    data_dir = Path(args.data_dir)
    interval_dir = data_dir / args.interval

    # --- Construct common arguments string FOR optimize.py ---
    # This string will be appended to the command run by the worker
    common_args_list = [
        f"--opt-start {shlex.quote(args.start_date)}",
        f"--opt-end {shlex.quote(args.end_date)}",
        f"--commission {args.commission}",
        f"--balance {args.balance}",
        f"--metric {shlex.quote(args.metric)}",
        f"--config {shlex.quote(args.config)}",  # Pass config path
        f"--processes {args.processes}",  # Pass inner processes count
    ]
    # Add boolean flags if set
    if args.save_details:
        common_args_list.append("--save-details")
    if args.apply_atr_filter:
        common_args_list.append("--apply-atr-filter")
    if args.apply_seasonality_filter:
        common_args_list.append("--apply-seasonality-filter")

    # Add arguments with values if they are provided
    if args.details_file:
        common_args_list.append(f"--details-file {shlex.quote(args.details_file)}")
    if args.atr_filter_period:
        common_args_list.append(f"--atr-filter-period {args.atr_filter_period}")
    if args.atr_filter_multiplier:
        common_args_list.append(f"--atr-filter-multiplier {args.atr_filter_multiplier}")
    if args.atr_filter_sma_period:
        common_args_list.append(f"--atr-filter-sma-period {args.atr_filter_sma_period}")
    if args.allowed_trading_hours_utc:
        common_args_list.append(
            f"--allowed-trading-hours-utc {shlex.quote(args.allowed_trading_hours_utc)}"
        )
    if args.apply_seasonality_to_symbols:
        common_args_list.append(
            f"--apply-seasonality-to-symbols {shlex.quote(args.apply_seasonality_to_symbols)}"
        )

    common_args_str_for_optimize_py = " ".join(common_args_list)
    logger.debug(
        f"Common args string for optimize.py: {common_args_str_for_optimize_py}"
    )

    # --- Prepare list of jobs ---
    tasks_to_submit = []
    total_potential_jobs = len(strategies) * len(symbols)

    logger.info(f"Preparing {total_potential_jobs} potential optimization jobs...")
    for strategy in strategies:
        for symbol in symbols:
            file_path = interval_dir / f"{symbol}_{args.interval}.csv"
            # We now rely on optimize.py (via check-data-file) to handle missing files
            # We will submit all jobs regardless of file existence here.
            # if not file_path.is_file():
            #     logger.warning(
            #         f"--- Skipping Strategy: {strategy}, Symbol: {symbol} - File not found: {file_path} ---"
            #     )
            #     continue

            tasks_to_submit.append(
                (
                    strategy,
                    symbol,
                    str(file_path),  # Pass file path even if it doesn't exist yet
                    common_args_str_for_optimize_py,  # Pass the constructed string
                )
            )

    actual_jobs_count = len(tasks_to_submit)
    if actual_jobs_count == 0:
        # This case should ideally not be hit if we submit all jobs,
        # but kept as a safeguard if strategies/symbols lists are empty.
        logger.error("No optimization jobs to run (check strategies/symbols). Exiting.")
        return

    logger.info(f"Submitting {actual_jobs_count} optimization jobs to the pool.")

    # Add job index and total count to each task
    jobs_with_indices = [
        task + (idx + 1, actual_jobs_count) for idx, task in enumerate(tasks_to_submit)
    ]

    # --- Run jobs in parallel ---
    # Use the --processes argument passed to this script for the pool size
    pool_processes = args.processes
    logger.info(f"Starting parallel optimization using {pool_processes} processes...")

    results = []
    try:
        with Pool(processes=pool_processes) as pool:
            # Use imap_unordered for potentially better memory usage and responsiveness
            for result in pool.imap_unordered(
                run_single_optimization, jobs_with_indices
            ):
                results.append(result)
                # Log completion (can be out of order)
                job_desc, ret_code, _, _ = result
                status = "succeeded" if ret_code == 0 else f"failed (code: {ret_code})"
                logger.info(f"Completed: {job_desc} - Status: {status}")

    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received. Terminating running jobs...")
        # Pool context manager handles termination
    except Exception as e:
        logger.error(f"An error occurred during pool processing: {e}", exc_info=True)

    # --- Summarize Results ---
    successful_jobs = sum(1 for _, ret_code, _, _ in results if ret_code == 0)
    failed_jobs = actual_jobs_count - successful_jobs

    logger.info("--- Batch Optimization Summary ---")
    logger.info(f"Total jobs submitted: {actual_jobs_count}")
    logger.info(f"Successful jobs: {successful_jobs}")
    logger.info(f"Failed jobs: {failed_jobs}")

    if failed_jobs > 0:
        logger.warning("Review logs for details on failed optimization runs.")

    logger.info("--- Batch optimization script finished. ---")


if __name__ == "__main__":
    main()
