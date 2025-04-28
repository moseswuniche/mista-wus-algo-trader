import pandas as pd
import numpy as np
import itertools
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Iterator, Tuple, Optional, cast
import logging
import collections.abc # Needed for recursive update check
import math # Added for calculating combinations

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

# --- Constants for Optimization Logging ---
PROGRESS_LOG_INTERVAL = 50 # Log progress every N combinations

def load_param_grid_from_config(config_path: str, symbol: str, strategy_class_name: str) -> Dict[str, List[Any]]:
    """Loads the parameter grid for a specific symbol and strategy from the YAML config."""
    config_file = Path(config_path)
    if not config_file.is_file():
        logger.error(f"Optimization config file not found: {config_path}")
        raise FileNotFoundError(f"Optimization config file not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            full_config = cast(Optional[Dict[str, Any]], yaml.safe_load(f))
            if full_config is None:
                full_config = {}
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

    # Cast the final return value, ensuring it's the expected specific type
    # This assumes validation confirms the structure fits Dict[str, List[Any]]
    validated_grid = cast(Dict[str, List[Any]], strategy_grid)

    # Basic validation: ensure values are lists
    for key, value in validated_grid.items():
        if not isinstance(value, list):
            logger.error(f"Invalid format for parameter '{key}' in {symbol}/{strategy_class_name}. Expected a list, got {type(value)}.")
            raise ValueError(f"Invalid format for parameter '{key}' in {symbol}/{strategy_class_name}. Expected a list, got {type(value)}.")
        if strategy_class_name == "LongShortStrategy" and key in ["return_thresh_low", "return_thresh_high"]:
             try:
                 validated_grid[key] = [float(v) for v in value]
             except ValueError:
                  logger.error(f"Invalid float value found for {key} in {symbol}/{strategy_class_name}.")
                  raise ValueError(f"Invalid float value found for {key} in {symbol}/{strategy_class_name}.")

    logger.info(f"Loaded parameter grid for {symbol} / {strategy_class_name} from {config_path}")
    return validated_grid

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
    commission_bps: float = 0.0,
    initial_balance: float = 10000.0
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
        initial_balance: Starting balance for backtests.

    Returns:
        A tuple containing: (best_params, best_result_dict) or (None, None) if error.
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

    # --- Calculate Total Combinations for Progress Logging --- 
    total_combinations = 1
    for param_values in param_grid.values():
         # Check for valid list and non-empty
         if isinstance(param_values, list) and param_values:
              total_combinations *= len(param_values)
         else:
              # Handle empty or invalid param lists if necessary, or assume valid based on loader
              logger.warning(f"Invalid or empty parameter list found in grid for {strategy_class_name}. Calculation might be inaccurate.")
              total_combinations = 0 # Or handle differently
              break 
              
    # Ensure total_combinations is not zero if loop completed
    if total_combinations == 1 and not any(param_grid.values()): # Check if grid was actually empty
         total_combinations = 0
         
    logger.info(f"--- Starting Optimization for {strategy_short_name} on {symbol} --- ")
    logger.info(f"Total parameter combinations to test: {total_combinations}")
    logger.info(f"Optimizing metric: {optimization_metric}, Commission: {commission_bps} bps, Initial Balance: {initial_balance}")
    
    param_generator = generate_param_combinations(param_grid, strategy_class_name)

    best_params = None
    best_result_dict = None # Rename variable for clarity
    
    # --- Determine if minimizing or maximizing the metric --- 
    metrics_to_minimize = ['max_drawdown']
    minimize = optimization_metric in metrics_to_minimize
    best_metric_value = float('inf') if minimize else -float('inf')
    comparison_operator = np.less if minimize else np.greater # Use np functions for comparison

    logger.info(f"Optimization mode: {'Minimizing' if minimize else 'Maximizing'} {optimization_metric}")

    count = 0
    results_list = [] # Store all results for detailed saving

    for params_combo in param_generator:
        count += 1
        # Separate strategy params from SL/TP/TSL params
        strategy_params = params_combo.copy()
        sl_pct = strategy_params.pop('stop_loss_pct', None)
        tp_pct = strategy_params.pop('take_profit_pct', None)
        tsl_pct = strategy_params.pop('trailing_stop_loss_pct', None)
        
        logger.debug(f"Testing combo {count}/{total_combinations}: StratP={strategy_params}, SL={sl_pct}, TP={tp_pct}, TSL={tsl_pct}")
        try:
            strategy_instance = strategy_class(**strategy_params)
            result_dict = run_backtest(
                data=data, 
                strategy=strategy_instance, 
                units=trade_units, 
                initial_balance=initial_balance,
                commission_bps=commission_bps,
                stop_loss_pct=sl_pct,
                take_profit_pct=tp_pct,
                trailing_stop_loss_pct=tsl_pct
            )
            
            full_params_tested = params_combo 
            # Add params to the result dict before appending to list
            result_dict_with_params = {'params': full_params_tested, **result_dict} 
            results_list.append(result_dict_with_params)

            current_metric_value = result_dict.get(optimization_metric)

            if current_metric_value is None or pd.isna(current_metric_value):
                 logger.warning(f"Metric '{optimization_metric}' missing or NaN for params {full_params_tested}. Skipping.")
                 continue

            # Compare using the chosen operator (np.less or np.greater)
            if comparison_operator(current_metric_value, best_metric_value):
                best_metric_value = current_metric_value
                best_params = full_params_tested 
                best_result_dict = result_dict # Store the result dict without params
                # Log the parameters *without* SL/TP/TSL if they are None, for brevity
                log_params = {k: v for k, v in best_params.items() if v is not None}
                logger.info(f"*** New best result (Combo {count}/{total_combinations}): {optimization_metric} = {best_metric_value:.4f} with params {log_params} ***")

        except Exception as e:
            logger.error(f"Error during backtest for params {full_params_tested}: {e}", exc_info=True)
            continue
            
        if total_combinations > 0 and count % PROGRESS_LOG_INTERVAL == 0:
            logger.info(f"Progress: Tested {count} / {total_combinations} combinations...")
            
    logger.info(f"--- Optimization Finished for {strategy_short_name} on {symbol} --- ")
    logger.info(f"Tested {count} parameter combinations.")
    if best_params:
        printable_best_params = adjust_params_for_printing(best_params, strategy_class_name)
        logger.info(f"Best {optimization_metric}: {best_metric_value:.4f}")
        logger.info(f"Best parameters (incl. SL/TP/TSL): {printable_best_params}")
    else:
        logger.warning("No valid results found during optimization.")

    # Return the best params (full combo), its result dict, and the full list of results
    return best_params, best_result_dict, results_list

def adjust_params_for_printing(params: Dict[str, Any], strategy_class_name: str) -> Dict[str, Any]:
    printable_params = params.copy()
    if strategy_class_name == "LongShortStrategy":
        if 'return_thresh' in printable_params:
            rt = printable_params.pop('return_thresh')
            printable_params['return_thresh_low'] = rt[0]
            printable_params['return_thresh_high'] = rt[1]
        if 'volume_thresh' in printable_params:
             vt = printable_params.pop('volume_thresh')
             printable_params['volume_thresh_low'] = vt[0]
             printable_params['volume_thresh_high'] = vt[1]
    # Add adjustments for other strategies if needed
    return printable_params

def save_best_params(output_config_path: str, symbol: str, strategy_class_name: str, best_params: Dict[str, Any]):
    """Saves the best parameters found (incl. SL/TP/TSL) to the specified YAML output file."""
    if not best_params:
        logger.warning("No best parameters found to save.")
        return
        
    output_file = Path(output_config_path)
    output_file.parent.mkdir(parents=True, exist_ok=True) 

    full_results = {}
    if output_file.is_file():
        try:
            with open(output_file, 'r') as f:
                full_results = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error reading existing results file {output_file}: {e}. Will overwrite.")
            full_results = {}
            
    # Adjust LongShortStrategy params back to low/high format before saving
    params_to_save = adjust_params_for_printing(best_params, strategy_class_name)

    # Update nested dictionary structure
    if symbol not in full_results:
        full_results[symbol] = {}
    full_results[symbol][strategy_class_name] = params_to_save

    try:
        with open(output_file, 'w') as f:
            yaml.dump(full_results, f, indent=2, default_flow_style=False)
        logger.info(f"Best parameters saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error writing best parameters to {output_file}: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Strategy Optimization")
    parser.add_argument("-s", "--strategy", type=str, required=True,
                        choices=list(STRATEGY_MAP.keys()),
                        help="Short name of the strategy to optimize.")
    parser.add_argument("--symbol", type=str, required=True,
                        help="Trading symbol (e.g., BTCUSDT). Optimization is symbol-specific.")
    parser.add_argument("--config", type=str, default="config/optimize_params.yaml",
                        help="Path to the optimization parameters YAML config file.")
    parser.add_argument("--file", type=str, required=True, 
                        help="Path to the historical data CSV file.")
    parser.add_argument("--opt-start", type=str, default=None,
                        help="Optional start date (YYYY-MM-DD) for the optimization period.")
    parser.add_argument("--opt-end", type=str, default=None,
                        help="Optional end date (YYYY-MM-DD) for the optimization period.")
    parser.add_argument("-u", "--units", type=float, default=1.0,
                        help="Amount/Units of asset to trade.")
    parser.add_argument("--metric", type=str, default="cumulative_profit", 
                        choices=['cumulative_profit', 'final_balance', 'sharpe_ratio', 'profit_factor', 'max_drawdown', 'win_rate'], 
                        help="Metric to optimize for. Default: cumulative_profit")
    parser.add_argument("--commission", type=float, default=10.0,
                        help="Commission fee in basis points (e.g., 10 for 0.1%). Default: 10.")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance for backtests.")
    parser.add_argument("--output-params", type=str, default="config/best_params.yaml", 
                        help="Output YAML file to save the best parameters.")
    parser.add_argument("--save-details", action="store_true", 
                        help="Save detailed results of all combinations to CSV.")
    parser.add_argument("--details-file", type=str, default=None, 
                        help="Optional path for detailed CSV results (defaults to results/optimization/...). Ignored if --save-details is not set.")
    parser.add_argument("--log", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level. Default: INFO")

    args = parser.parse_args()

    # Set logging level from args
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    try:
        logger.info(f"Starting optimization run with args: {args}")
        # Load FULL dataset first using the required --file argument
        full_data = load_csv_data(args.file, symbol=args.symbol)
        full_data.sort_index(inplace=True)
        logger.info(f"Loaded full dataset: {len(full_data)} rows from {full_data.index.min()} to {full_data.index.max()}")

        # Slice data for optimization period if start/end dates are provided
        opt_data = full_data
        slice_info = "full dataset"
        if args.opt_start or args.opt_end:
            start_slice = pd.to_datetime(args.opt_start) if args.opt_start else None
            end_slice = pd.to_datetime(args.opt_end) if args.opt_end else None
            try:
                opt_data = full_data.loc[start_slice:end_slice].copy()
                slice_info = f"from {opt_data.index.min()} to {opt_data.index.max()}"
                if opt_data.empty:
                     raise ValueError(f"No data found within the specified optimization period: {args.opt_start} - {args.opt_end}")
            except Exception as e:
                logger.error(f"Error slicing data for optimization period ({args.opt_start} - {args.opt_end}): {e}", exc_info=True)
                # Decide how to handle: exit or proceed with full data?
                # For safety, let's exit if slicing fails or results in empty data
                raise ValueError(f"Failed to slice data for optimization: {e}") 

        logger.info(f"Using data slice for optimization: {len(opt_data)} rows ({slice_info})")
        
        # Run Optimization on the potentially sliced data
        best_params, best_result_dict, all_results_list = optimize_strategy(
            strategy_short_name=args.strategy,
            config_path=args.config,
            symbol=args.symbol,
            data=opt_data, # Pass the potentially sliced data
            trade_units=args.units,
            optimization_metric=args.metric,
            commission_bps=args.commission,
            initial_balance=args.balance
        )

        # --- Save Best Parameters --- 
        if best_params:
            strategy_class = STRATEGY_MAP[args.strategy]
            strategy_class_name = strategy_class.__name__
            save_best_params(
                output_config_path=args.output_params,
                symbol=args.symbol,
                strategy_class_name=strategy_class_name,
                best_params=best_params
            )
            logger.info(f"Best parameters saved to: {args.output_params}")
        else:
            logger.warning("Optimization did not yield best parameters. Results file not updated.")
            
        if best_result_dict and isinstance(best_result_dict.get('performance_summary'), pd.DataFrame):
             logger.info("\n--- Performance Summary for Best Parameters ---")
             summary_string = best_result_dict['performance_summary'].to_string()
             logger.info(f"\n{summary_string}")

        # --- Save Detailed Results if requested --- 
        if args.save_details:
            if all_results_list:
                try:
                    # Define default filename if not provided
                    if args.details_file is None:
                        details_dir = Path("results/optimization")
                        details_dir.mkdir(parents=True, exist_ok=True)
                        filename = f"{args.strategy}_{args.symbol}_optimize_details_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        details_path = details_dir / filename
                    else:
                        details_path = Path(args.details_file)
                        details_path.parent.mkdir(parents=True, exist_ok=True)

                    # Flatten the results for CSV
                    flat_results = []
                    for res in all_results_list:
                        row = res['params'].copy() # Start with parameters
                        # Add other results, prefixing to avoid name collisions if needed
                        for k, v in res.items():
                            if k != 'params' and k!= 'performance_summary' and k != 'equity_curve': # Exclude complex objects
                                 row[f'result_{k}'] = v
                        flat_results.append(row)
                        
                    results_df = pd.DataFrame(flat_results)
                    results_df.to_csv(details_path, index=False)
                    logger.info(f"Saved detailed optimization results to: {details_path}")
                    
                except Exception as e:
                    logger.error(f"Error saving detailed optimization results: {e}", exc_info=True)
            else:
                logger.warning("Detailed results requested, but no results were generated during optimization.")

    except FileNotFoundError as fnf:
        logger.error(f"File Not Found: {fnf}", exc_info=True)
    except ValueError as ve:
        logger.error(f"Configuration or Value Error: {ve}", exc_info=True)
    except IOError as ioe:
        logger.error(f"File Reading Error: {ioe}", exc_info=True)
    except Exception as e:
        import traceback
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
    
    logger.info("--- Optimization script finished. ---") 