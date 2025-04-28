import pandas as pd
import numpy as np
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, cast
import logging

# Assuming strategies are accessible via this import path
from .strategies import (
    Strategy,
    LongShortStrategy,
    MovingAverageCrossoverStrategy,
    RsiMeanReversionStrategy,
    BollingerBandReversionStrategy
)
from .backtest import run_backtest
from .data_utils import load_csv_data

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map strategy short names (used in args) to classes
STRATEGY_CLASS_MAP = {
    "LongShort": LongShortStrategy,
    "MACross": MovingAverageCrossoverStrategy,
    "RSIReversion": RsiMeanReversionStrategy,
    "BBReversion": BollingerBandReversionStrategy
}

# --- Parameter Loading Function (Similar to optimize.py but for specific values) ---
def load_best_params_from_config(config_path: str, symbol: str, strategy_class_name: str) -> Optional[Dict[str, Any]]:
    """Loads the specific best parameters for a symbol/strategy from a YAML config."""
    config_file = Path(config_path)
    if not config_file.is_file():
        logger.error(f"Parameter config file not found: {config_path}")
        return None

    try:
        with open(config_file, 'r') as f:
            full_config = yaml.safe_load(f)
            if full_config is None: 
                 logger.warning(f"Parameter config file is empty: {config_path}")
                 return None
    except Exception as e:
        logger.error(f"Error reading parameter config file {config_path}: {e}", exc_info=True)
        return None

    if not isinstance(full_config, dict):
        logger.error(f"Invalid YAML format in {config_path}. Expected a top-level dictionary.")
        return None

    symbol_config = full_config.get(symbol)
    if not symbol_config or not isinstance(symbol_config, dict):
        logger.error(f"Symbol '{symbol}' not found or invalid format in config file: {config_path}")
        return None

    strategy_params = symbol_config.get(strategy_class_name)
    if not strategy_params or not isinstance(strategy_params, dict):
        logger.error(f"Strategy '{strategy_class_name}' parameters not found or invalid format for symbol '{symbol}' in config: {config_path}")
        return None

    logger.info(f"Loaded parameters for {symbol} / {strategy_class_name} from {config_path}: {strategy_params}")
    
    # --- Adjust for LongShortStrategy's tuple parameters --- 
    if strategy_class_name == "LongShortStrategy":
        # Check if the split keys exist from the optimization save
        if 'return_thresh_low' in strategy_params and 'return_thresh_high' in strategy_params:
            strategy_params['return_thresh'] = (
                strategy_params.pop('return_thresh_low'), 
                strategy_params.pop('return_thresh_high')
            )
        elif 'return_thresh' not in strategy_params:
             logger.error(f"Missing return threshold parameters for LongShortStrategy in {config_path}")
             return None # Or raise error
             
        if 'volume_thresh_low' in strategy_params and 'volume_thresh_high' in strategy_params:
            strategy_params['volume_thresh'] = (
                strategy_params.pop('volume_thresh_low'), 
                strategy_params.pop('volume_thresh_high')
            )
        elif 'volume_thresh' not in strategy_params:
             logger.error(f"Missing volume threshold parameters for LongShortStrategy in {config_path}")
             return None # Or raise error
             
    return strategy_params

# --- Main Forward Testing Function ---
def run_forward_test(
    strategy_short_name: str,
    param_config_path: str,
    symbol: str,
    data: pd.DataFrame,
    fwd_test_start: str,
    fwd_test_end: Optional[str],
    trade_units: float,
    commission_bps: float
):
    """Runs a backtest on a specified forward-testing period using pre-defined parameters."""
    logger.info(f"--- Starting Forward Test for {strategy_short_name} on {symbol} --- ")
    logger.info(f"Parameters loading from: {param_config_path}")
    logger.info(f"Forward Test Period: {fwd_test_start} to {fwd_test_end or 'End of Data'}")

    # 1. Get Strategy Class
    if strategy_short_name not in STRATEGY_CLASS_MAP:
        logger.error(f"Unknown strategy name: {strategy_short_name}. Available: {list(STRATEGY_CLASS_MAP.keys())}")
        return
    strategy_class = STRATEGY_CLASS_MAP[strategy_short_name]
    strategy_class_name = strategy_class.__name__

    # 2. Load Best Parameters
    best_params = load_best_params_from_config(param_config_path, symbol, strategy_class_name)
    if best_params is None:
        logger.error(f"Could not load best parameters for {symbol}/{strategy_class_name}. Aborting forward test.")
        return
        
    # 3. Instantiate Strategy
    try:
        strategy_instance = strategy_class(**best_params)
        logger.info(f"Instantiated strategy {strategy_class_name} with parameters: {best_params}")
    except Exception as e:
        logger.error(f"Error instantiating strategy {strategy_class_name} with params {best_params}: {e}", exc_info=True)
        return

    # 4. Slice Data for Forward Test Period
    try:
        fwd_start_dt = pd.to_datetime(fwd_test_start)
        if fwd_test_end:
            fwd_end_dt = pd.to_datetime(fwd_test_end)
            forward_data = data.loc[fwd_start_dt:fwd_end_dt].copy()
        else:
            forward_data = data.loc[fwd_start_dt:].copy()
            
        if forward_data.empty:
             logger.error(f"No data found for the specified forward test period ({fwd_test_start} to {fwd_test_end or 'end'}).")
             return
             
        logger.info(f"Data sliced for forward test: {len(forward_data)} rows from {forward_data.index.min()} to {forward_data.index.max()}")
    except Exception as e:
        logger.error(f"Error slicing data for forward test period ({fwd_test_start} - {fwd_test_end}): {e}", exc_info=True)
        return

    # 5. Run Backtest on Forward Data
    logger.info("Running backtest on the forward test data slice...")
    try:
        fwd_results = run_backtest(
            data=forward_data,
            strategy=strategy_instance,
            units=trade_units,
            commission_bps=commission_bps
        )
        
        # Display Forward Test Results
        logger.info("--- Forward Test Results --- ")
        if fwd_results:
             logger.info(f"Cumulative Profit: {fwd_results.get('cumulative_profit'):.4f}")
             logger.info(f"Final Balance: {fwd_results.get('final_balance'):.2f}")
             logger.info(f"Total Trades: {fwd_results.get('total_trades')}")
             logger.info(f"Win Rate: {fwd_results.get('win_rate'):.2f}%")
             summary_df = fwd_results.get('performance_summary')
             if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
                  logger.info("\n--- Forward Test Trade Summary ---")
                  logger.info(f"\n{summary_df.to_string()}")
             else:
                  logger.info("No trades executed during the forward test period.")
        else:
            logger.warning("Backtest function returned no results.")
            
    except Exception as e:
        logger.error(f"Error running backtest during forward test: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Simulated Forward Test using Optimized Parameters")
    parser.add_argument("-s", "--strategy", type=str, required=True,
                        choices=list(STRATEGY_CLASS_MAP.keys()),
                        help="Short name of the strategy previously optimized.")
    parser.add_argument("--symbol", type=str, required=True,
                        help="Trading symbol (e.g., BTCUSDT) to load data and parameters for.")
    parser.add_argument("--param-config", type=str, default="config/best_params.yaml",
                        help="Path to the YAML file containing the best parameters.")
    parser.add_argument("-f", "--file", type=str, default=None,
                        help="Path to the FULL historical data CSV file. If None, attempts default path (e.g., data/SYMBOL_1d.csv).")
    parser.add_argument("--fwd-start", type=str, required=True,
                        help="Start date (YYYY-MM-DD) for the forward testing period.")
    parser.add_argument("--fwd-end", type=str, default=None,
                        help="Optional end date (YYYY-MM-DD) for the forward testing period. If None, uses data until the end.")
    parser.add_argument("-u", "--units", type=float, default=1.0,
                        help="Amount/Units of asset to trade.")
    parser.add_argument("--commission", type=float, default=10.0,
                        help="Commission fee in basis points (e.g., 10 for 0.1%). Default: 10.")
    parser.add_argument("--log", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level. Default: INFO")

    args = parser.parse_args()

    # Set logging level
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    # Determine data file path (using daily '1d' as a common default if not specified)
    if args.file is None:
        args.file = f"data/{args.symbol}_1d.csv" # Assuming daily data is common for fwd test
        logger.info(f"Data file not specified, attempting default: {args.file}")
        # Add fallback for bitcoin.csv if needed?
        fallback_file = "src/trading_bots/bitcoin.csv" # Assuming this is daily BTC
        if not Path(args.file).is_file() and args.symbol == "BTCUSDT" and Path(fallback_file).is_file():
             args.file = fallback_file
             logger.info(f"Default file not found, using fallback: {args.file}")
             
    try:
        logger.info(f"Starting forward test run with args: {args}")
        # Load FULL dataset
        full_data = load_csv_data(args.file, symbol=args.symbol)
        full_data.sort_index(inplace=True)
        logger.info(f"Loaded full dataset: {len(full_data)} rows from {full_data.index.min()} to {full_data.index.max()}")

        # Run the forward test function
        run_forward_test(
            strategy_short_name=args.strategy,
            param_config_path=args.param_config,
            symbol=args.symbol,
            data=full_data, # Pass the full dataset
            fwd_test_start=args.fwd_start,
            fwd_test_end=args.fwd_end,
            trade_units=args.units,
            commission_bps=args.commission
        )

    except FileNotFoundError as fnf:
        logger.error(f"File Not Found: {fnf}", exc_info=True)
    except ValueError as ve:
        logger.error(f"Configuration or Value Error: {ve}", exc_info=True)
    except IOError as ioe:
        logger.error(f"File Reading Error: {ioe}", exc_info=True)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)

    logger.info("--- Forward test script finished. ---") 