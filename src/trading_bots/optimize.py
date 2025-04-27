import pandas as pd
import numpy as np
import itertools
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Iterator, Tuple, Optional
import logging
import collections.abc # Needed for recursive update check

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
STRATEGY_MAP = {
    "LongShort": LongShortStrategy,
    "MACross": MovingAverageCrossoverStrategy,
    "RSIReversion": RsiMeanReversionStrategy,
    "BBReversion": BollingerBandReversionStrategy
}

def load_param_grid_from_config(config_path: str, symbol: str, strategy_class_name: str) -> Dict[str, List[Any]]:
    """Loads the parameter grid for a specific symbol and strategy from the YAML config."""
    config_file = Path(config_path)
    if not config_file.is_file():
        logger.error(f"Optimization config file not found: {config_path}")
        raise FileNotFoundError(f"Optimization config file not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            full_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file {config_path}: {e}")
        raise ValueError(f"Error parsing YAML config file {config_path}: {e}")
    except Exception as e:
        logger.error(f"Error reading config file {config_path}: {e}")
        raise IOError(f"Error reading config file {config_path}: {e}")

    if not isinstance(full_config, dict):
        logger.error(f"Invalid YAML format in {config_path}. Expected a dictionary.")
        raise ValueError(f"Invalid YAML format in {config_path}. Expected a dictionary.")

    symbol_config = full_config.get(symbol)
    if not symbol_config or not isinstance(symbol_config, dict):
        logger.error(f"Symbol '{symbol}' not found or invalid format in config file: {config_path}")
        raise ValueError(f"Symbol '{symbol}' not found or invalid format in config file: {config_path}")

    strategy_grid = symbol_config.get(strategy_class_name)
    if not strategy_grid or not isinstance(strategy_grid, dict):
        logger.error(f"Strategy '{strategy_class_name}' not found or invalid format for symbol '{symbol}' in config: {config_path}")
        raise ValueError(f"Strategy '{strategy_class_name}' not found or invalid format for symbol '{symbol}' in config: {config_path}")

    # Basic validation: ensure values are lists
    for key, value in strategy_grid.items():
        if not isinstance(value, list):
            logger.error(f"Invalid format for parameter '{key}' in {symbol}/{strategy_class_name}. Expected a list, got {type(value)}.")
            raise ValueError(f"Invalid format for parameter '{key}' in {symbol}/{strategy_class_name}. Expected a list, got {type(value)}.")
        if strategy_class_name == "LongShortStrategy" and key in ["return_thresh_low", "return_thresh_high"]:
             try:
                 strategy_grid[key] = [float(v) for v in value]
             except ValueError:
                  logger.error(f"Invalid float value found for {key} in {symbol}/{strategy_class_name}.")
                  raise ValueError(f"Invalid float value found for {key} in {symbol}/{strategy_class_name}.")

    logger.info(f"Loaded parameter grid for {symbol} / {strategy_class_name} from {config_path}")
    return strategy_grid

def generate_param_combinations(param_grid: Dict[str, List[Any]], strategy_class_name: str) -> Iterator[Dict[str, Any]]:
    """Generates parameter combinations for grid search from a loaded grid."""
    param_names = list(param_grid.keys())
    value_lists = list(param_grid.values())

    for combo_values in itertools.product(*value_lists):
        params = dict(zip(param_names, combo_values))

        # --- Apply Strategy-Specific Constraints & Adjustments --- 
        if strategy_class_name == "MovingAverageCrossoverStrategy":
            if params['fast_period'] >= params['slow_period']:
                continue # Skip invalid combination
        elif strategy_class_name == "RsiMeanReversionStrategy":
            if params['oversold_threshold'] >= params['overbought_threshold']:
                continue # Skip invalid combination
        elif strategy_class_name == "LongShortStrategy":
            # Parameter names need adjustment for LongShortStrategy's __init__
            # Use np.round here if precise float ranges are needed and defined as such in YAML
            # or handle potential float inaccuracies from YAML load if necessary.
            # Assuming YAML stores them as intended floats now.
            rt_low = params.pop('return_thresh_low')
            rt_high = params.pop('return_thresh_high')
            vt_low = params.pop('volume_thresh_low')
            vt_high = params.pop('volume_thresh_high')
            params['return_thresh'] = (rt_low, rt_high)
            params['volume_thresh'] = (vt_low, vt_high)

        yield params

def optimize_strategy(
    strategy_short_name: str,
    config_path: str,
    symbol: str,
    data: pd.DataFrame,
    trade_units: float,
    optimization_metric: str = 'cumulative_profit',
    commission_bps: float = 0.0
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Performs grid search optimization for a given strategy.

    Args:
        strategy_short_name: The short name of the strategy to optimize (e.g., "MACross").
        config_path: Path to the YAML configuration file.
        symbol: The trading symbol (e.g., "BTCUSDT") to load config for.
        data: Historical data DataFrame.
        trade_units: Units to trade.
        optimization_metric: The key in the backtest result to maximize.
        commission_bps: Commission fee in basis points passed to backtester.

    Returns:
        A tuple containing: (best_params, best_result) or (None, None) if error.
    """
    if strategy_short_name not in STRATEGY_MAP:
        logger.error(f"Unknown strategy name: {strategy_short_name}. Available: {list(STRATEGY_MAP.keys())}")
        raise ValueError(f"Unknown strategy name: {strategy_short_name}. Available: {list(STRATEGY_MAP.keys())}")
    
    strategy_class = STRATEGY_MAP[strategy_short_name]
    strategy_class_name = strategy_class.__name__

    try:
        # Load the specific grid for this symbol and strategy
        param_grid = load_param_grid_from_config(config_path, symbol, strategy_class_name)
    except (FileNotFoundError, ValueError, IOError) as e:
        logger.error(f"Stopping optimization due to error loading parameter grid: {e}")
        return None, None

    param_generator = generate_param_combinations(param_grid, strategy_class_name)

    best_params = None
    best_result = None
    best_metric_value = -float('inf')
    
    count = 0
    results_list = []

    logger.info(f"\n--- Starting Optimization for {strategy_short_name} on {symbol} --- ")
    logger.info(f"Optimizing metric: {optimization_metric}, Commission: {commission_bps} bps")

    for params in param_generator:
        count += 1
        logger.debug(f"\nTesting combination {count}: {params}")
        try:
            strategy_instance = strategy_class(**params)
            result = run_backtest(data, strategy_instance, trade_units, commission_bps=commission_bps)
            results_list.append({'params': params, **result})

            current_metric_value = result.get(optimization_metric)

            if current_metric_value is None:
                logger.warning(f"Metric '{optimization_metric}' not found in backtest results for params {params}. Skipping.")
                continue
            
            if pd.isna(current_metric_value):
                 logger.warning(f"Metric '{optimization_metric}' is NaN for params {params}. Skipping.")
                 continue

            if current_metric_value > best_metric_value:
                best_metric_value = current_metric_value
                best_params = params
                best_result = result
                logger.info(f"*** New best result found (Combo {count}): {optimization_metric} = {best_metric_value:.4f} ***")

        except Exception as e:
            logger.error(f"Error during backtest for params {params}: {e}", exc_info=True)
            continue
            
    logger.info(f"\n--- Optimization Finished for {strategy_short_name} on {symbol} --- ")
    logger.info(f"Tested {count} parameter combinations.")
    if best_params:
        if strategy_class_name == "LongShortStrategy":
            printable_params = best_params.copy()
            rt = printable_params.pop('return_thresh')
            vt = printable_params.pop('volume_thresh')
            printable_params['return_thresh_low'] = rt[0]
            printable_params['return_thresh_high'] = rt[1]
            printable_params['volume_thresh_low'] = vt[0]
            printable_params['volume_thresh_high'] = vt[1]
        else:
            printable_params = best_params

        logger.info(f"Best {optimization_metric}: {best_metric_value:.4f}")
        logger.info(f"Best parameters: {printable_params}")
    else:
        logger.warning("No valid results found during optimization.")

    return best_params, best_result

def save_best_params(output_config_path: str, symbol: str, strategy_class_name: str, best_params: Dict[str, Any]):
    """Saves the best parameters found to the specified YAML output file."""
    output_file = Path(output_config_path)
    output_file.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    full_results = {}
    if output_file.is_file():
        try:
            with open(output_file, 'r') as f:
                full_results = yaml.safe_load(f)
                if full_results is None: # Handle empty file
                    full_results = {}
                if not isinstance(full_results, dict):
                     logger.warning(f"Existing output file {output_config_path} has invalid format. Overwriting.")
                     full_results = {}
        except Exception as e:
            logger.error(f"Error reading existing output file {output_config_path}: {e}. Will attempt to overwrite.", exc_info=True)
            full_results = {}

    # Navigate or create structure
    if symbol not in full_results:
        full_results[symbol] = {}
    if not isinstance(full_results[symbol], dict):
        logger.warning(f"Overwriting non-dictionary entry for symbol {symbol} in output file.")
        full_results[symbol] = {}
        
    # Prepare params for saving (handle LongShort tuple split)
    params_to_save = best_params.copy()
    if strategy_class_name == "LongShortStrategy":
        if 'return_thresh' in params_to_save:
            rt_low, rt_high = params_to_save.pop('return_thresh')
            params_to_save['return_thresh_low'] = rt_low
            params_to_save['return_thresh_high'] = rt_high
        if 'volume_thresh' in params_to_save:
             vt_low, vt_high = params_to_save.pop('volume_thresh')
             params_to_save['volume_thresh_low'] = vt_low
             params_to_save['volume_thresh_high'] = vt_high
             
    # Store the best params for this specific strategy under the symbol
    full_results[symbol][strategy_class_name] = params_to_save
    logger.info(f"Saving best parameters for {symbol} / {strategy_class_name} to {output_config_path}")

    try:
        with open(output_file, 'w') as f:
            yaml.dump(full_results, f, default_flow_style=None, sort_keys=False)
        logger.info(f"Best parameters saved successfully to {output_config_path}.")
    except Exception as e:
        logger.error(f"Error writing best parameters to {output_config_path}: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Trading Strategy Parameters")
    parser.add_argument("-s", "--strategy", type=str, required=True,
                        choices=list(STRATEGY_MAP.keys()),
                        help="Short name of the strategy to optimize.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", 
                        help="Trading symbol (e.g., BTCUSDT, XRPUSDT) used for loading data and config.")
    parser.add_argument("-f", "--file", type=str, default=None,
                        help="Path to the historical data CSV file. If None, attempts default path based on symbol (e.g., data/BTCUSDT_1h.csv). Needs implementation.")
    parser.add_argument("--config", type=str, default="config/optimize_params.yaml",
                        help="Path to the optimization parameters YAML config file.")
    parser.add_argument("-u", "--units", type=float, default=1.0,
                        help="Amount/Units of asset to trade.")
    parser.add_argument("-m", "--metric", type=str, default="cumulative_profit",
                        choices=['cumulative_profit', 'final_balance', 'total_trades', 'win_rate'],
                        help="Metric to optimize (maximize).")
    parser.add_argument("--commission", type=float, default=10.0,
                        help="Commission fee in basis points (e.g., 10 for 0.1%). Default: 10.")
    parser.add_argument("--log", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level. Default: INFO")
    parser.add_argument("--output-config", type=str, default="config/best_params.yaml",
                        help="Path to save the optimized parameters YAML file.")

    args = parser.parse_args()

    # Set logging level from args
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    if args.file is None:
        args.file = f"data/{args.symbol}_1h.csv"
        logger.info(f"Data file not specified, attempting default: {args.file}")
        fallback_file = "src/trading_bots/bitcoin.csv"
        if not Path(args.file).is_file() and args.symbol == "BTCUSDT" and Path(fallback_file).is_file():
             args.file = fallback_file
             logger.info(f"Default file not found, using fallback: {args.file}")
        
    try:
        logger.info(f"Starting optimization run with args: {args}")
        data = load_csv_data(args.file, symbol=args.symbol)
        
        data.sort_index(inplace=True)

        best_params, best_result = optimize_strategy(
            strategy_short_name=args.strategy,
            config_path=args.config,
            symbol=args.symbol,
            data=data,
            trade_units=args.units,
            optimization_metric=args.metric,
            commission_bps=args.commission
        )

        # --- Save Best Parameters --- 
        if best_params:
            strategy_class = STRATEGY_MAP.get(args.strategy)
            if strategy_class:
                save_best_params(
                    output_config_path=args.output_config,
                    symbol=args.symbol,
                    strategy_class_name=strategy_class.__name__,
                    best_params=best_params
                )
            else:
                 logger.error(f"Could not find strategy class for '{args.strategy}' to save results.")
        else:
            logger.warning("Optimization did not yield best parameters. Results file not updated.")
            
        if best_result and isinstance(best_result.get('performance_summary'), pd.DataFrame):
             logger.info("\n--- Performance Summary for Best Parameters ---")
             summary_string = best_result['performance_summary'].to_string()
             logger.info(f"\n{summary_string}")

    except FileNotFoundError as fnf:
        logger.error(f"File Not Found: {fnf}", exc_info=True)
    except ValueError as ve:
        logger.error(f"Configuration or Value Error: {ve}", exc_info=True)
    except IOError as ioe:
        logger.error(f"File Reading Error: {ioe}", exc_info=True)
    except Exception as e:
        import traceback
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
    
    logger.info("Optimization script finished.") 