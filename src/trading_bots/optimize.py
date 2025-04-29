import pandas as pd
import numpy as np
import itertools
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Iterator, Tuple, Optional, cast
import logging
import collections.abc  # Needed for recursive update check
import math  # Added for calculating combinations
import string  # Added for filename sanitization
from multiprocessing import Pool, cpu_count  # Added for parallel backtesting

# Assuming strategies are accessible via this import path
from .strategies import (
    Strategy,
    LongShortStrategy,
    MovingAverageCrossoverStrategy,
    RsiMeanReversionStrategy,
    BollingerBandReversionStrategy,
)
from .backtest import run_backtest
from .data_utils import load_csv_data

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Map strategy short names (used in args) to classes
STRATEGY_MAP = {
    "LongShort": LongShortStrategy,
    "MACross": MovingAverageCrossoverStrategy,
    "RSIReversion": RsiMeanReversionStrategy,
    "BBReversion": BollingerBandReversionStrategy,
}

# --- Constants for Optimization Logging ---
PROGRESS_LOG_INTERVAL = 50  # Log progress every N combinations
DEFAULT_OPT_PROCESSES = (
    6  # Optimized default for M3 Pro (typically 6 Performance cores)
)


def load_param_grid_from_config(
    config_path: str, symbol: str, strategy_class_name: str
) -> Dict[str, List[Any]]:
    """Loads the parameter grid for a specific symbol and strategy from the YAML config."""
    config_file = Path(config_path).resolve()
    logger.debug(f"Attempting to load param grid from resolved path: {config_file}")
    if not config_file.is_file():
        logger.error(
            f"Optimization config file not found at resolved path: {config_file}"
        )
        raise FileNotFoundError(
            f"Optimization config file not found at resolved path: {config_file}"
        )

    try:
        with open(config_file, "r") as f:
            full_config = cast(Optional[Dict[str, Any]], yaml.safe_load(f))
            if full_config is None:
                full_config = {}
            logger.debug(f"Loaded full config. Keys: {list(full_config.keys())}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file {config_path}: {e}")
        raise ValueError(f"Error parsing YAML config file {config_path}: {e}")
    except Exception as e:
        logger.error(f"Error reading config file {config_path}: {e}")
        raise IOError(f"Error reading config file {config_path}: {e}")

    if not isinstance(full_config, dict):
        logger.error(f"Invalid YAML format in {config_path}. Expected a dictionary.")
        raise ValueError(
            f"Invalid YAML format in {config_path}. Expected a dictionary."
        )

    logger.debug(f"Looking for symbol '{symbol}' in config...")
    symbol_config = full_config.get(symbol)
    if not symbol_config or not isinstance(symbol_config, dict):
        logger.error(
            f"Symbol '{symbol}' not found or invalid format in config file: {config_path}"
        )
        raise ValueError(
            f"Symbol '{symbol}' not found or invalid format in config file: {config_path}"
        )
    logger.debug(f"Found symbol config. Keys: {list(symbol_config.keys())}")

    logger.debug(
        f"Looking for strategy '{strategy_class_name}' under symbol '{symbol}'..."
    )
    strategy_grid = symbol_config.get(strategy_class_name)
    if not strategy_grid or not isinstance(strategy_grid, dict):
        logger.error(
            f"Strategy '{strategy_class_name}' not found or invalid format for symbol '{symbol}' in config: {config_path}"
        )
        raise ValueError(
            f"Strategy '{strategy_class_name}' not found or invalid format for symbol '{symbol}' in config: {config_path}"
        )
    logger.debug("Found strategy grid.")

    # Cast the final return value, ensuring it's the expected specific type
    # This assumes validation confirms the structure fits Dict[str, List[Any]]
    validated_grid = cast(Dict[str, List[Any]], strategy_grid)

    # Basic validation: ensure values are lists
    for key, value in validated_grid.items():
        if not isinstance(value, list):
            logger.error(
                f"Invalid format for parameter '{key}' in {symbol}/{strategy_class_name}. Expected a list, got {type(value)}."
            )
            raise ValueError(
                f"Invalid format for parameter '{key}' in {symbol}/{strategy_class_name}. Expected a list, got {type(value)}."
            )
        if strategy_class_name == "LongShortStrategy" and key in [
            "return_thresh_low",
            "return_thresh_high",
        ]:
            try:
                validated_grid[key] = [float(v) for v in value]
            except ValueError:
                logger.error(
                    f"Invalid float value found for {key} in {symbol}/{strategy_class_name}."
                )
                raise ValueError(
                    f"Invalid float value found for {key} in {symbol}/{strategy_class_name}."
                )

    logger.info(
        f"Loaded parameter grid for {symbol} / {strategy_class_name} from {config_path}"
    )
    return validated_grid


def generate_param_combinations(
    param_grid: Dict[str, List[Any]], strategy_class_name: str
) -> Iterator[Dict[str, Any]]:
    """Generates parameter combinations for grid search from a loaded grid."""
    param_names = list(param_grid.keys())
    value_lists = list(param_grid.values())

    for combo_values in itertools.product(*value_lists):
        params = dict(zip(param_names, combo_values))

        # --- Apply Strategy-Specific Constraints & Adjustments ---
        if strategy_class_name == "MovingAverageCrossoverStrategy":
            if params["fast_period"] >= params["slow_period"]:
                continue  # Skip invalid combination
        elif strategy_class_name == "RsiMeanReversionStrategy":
            if params["oversold_threshold"] >= params["overbought_threshold"]:
                continue  # Skip invalid combination
        elif strategy_class_name == "LongShortStrategy":
            # Pop the source threshold values
            rt_low = params.pop("rsi_threshold_low")
            rt_high = params.pop("rsi_threshold_high")
            vt_low = params.pop("volume_z_thresh_low")
            vt_high = params.pop("volume_z_thresh_high")

            # Also pop the indicator parameters that are not direct constructor args
            params.pop("fast_ema_period", None)  # Use None default in case key varies
            params.pop("slow_ema_period", None)
            params.pop("rsi_period", None)

            # Add the expected tuple parameters for the constructor
            params["return_thresh"] = (rt_low, rt_high)
            params["volume_thresh"] = (vt_low, vt_high)

            # params dict now only contains keys expected by LongShortStrategy.__init__
            # (assuming it expects return_thresh and volume_thresh tuples)

        yield params


# <<< Helper Function for Filename Sanitization >>>
def sanitize_filename(filename: str) -> str:
    """Removes or replaces characters invalid for filenames."""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    sanitized = "".join(c if c in valid_chars else "_" for c in filename)
    return sanitized


# <<< Helper Function to Create Parameter String for Filename >>>
def create_param_string(params: Dict[str, Any]) -> str:
    """Creates a sorted, filename-safe string from parameter dictionary."""
    items = []
    for key, value in sorted(params.items()):  # Sort for consistency
        # Format value (handle None, floats, tuples)
        if value is None:
            value_str = "None"
        elif isinstance(value, float):
            value_str = f"{value:.4g}".replace(
                ".", "p"
            )  # Use general format, replace dot
        elif isinstance(value, tuple):
            # Format tuple elements individually
            value_str = "_".join(
                f"{v:.4g}".replace(".", "p") if isinstance(v, float) else str(v)
                for v in value
            )
        else:
            value_str = str(value)
        items.append(f"{key}_{value_str}")
    return sanitize_filename("_".join(items))


# --- Worker function for parallel backtesting ---
def run_backtest_for_params(args_dict: Dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Runs backtest for a single parameter set. Designed for multiprocessing.

    Args:
        args_dict: Dictionary containing keys like 'params', 'strategy_class',
                   'data', 'trade_units', 'commission_bps', 'initial_balance'.

    Returns:
        A tuple: (params, result_dict)
    """
    params = args_dict["params"]
    strategy_class = args_dict["strategy_class"]
    symbol = args_dict["symbol"]
    data = args_dict["data"]
    trade_units = args_dict["trade_units"]
    commission_bps = args_dict["commission_bps"]
    initial_balance = args_dict["initial_balance"]
    # --- Get optional SL/TP values from params, default to None ---
    stop_loss_pct = params.get("stop_loss_pct")
    take_profit_pct = params.get("take_profit_pct")
    trailing_stop_loss_pct = params.get("trailing_stop_loss_pct")
    # --- Remove SL/TP/TSL from strategy params before instantiation ---
    strategy_only_params = {
        k: v
        for k, v in params.items()
        if k not in ["stop_loss_pct", "take_profit_pct", "trailing_stop_loss_pct"]
    }

    try:
        # Instantiate the strategy with the filtered parameter combination
        strategy_instance = strategy_class(**strategy_only_params)

        # Log start including symbol
        logger.info(f"Running backtest for: {strategy_class.__name__} on {symbol}")

        # Run the backtest with correct args
        result_dict = run_backtest(
            strategy=strategy_instance,
            data=data,
            units=trade_units,  # Corrected arg name: units
            commission_bps=commission_bps,
            initial_balance=initial_balance,
            stop_loss_pct=stop_loss_pct,  # Pass extracted SL
            take_profit_pct=take_profit_pct,  # Pass extracted TP
            trailing_stop_loss_pct=trailing_stop_loss_pct,  # Pass extracted TSL
            # log_trades=False, # Removed invalid arg
        )
        return params, result_dict  # Return full params and results
    except Exception as e:
        logger.error(
            f"Exception during backtest for params {params}: {e}", exc_info=True
        )
        # Return params and an empty result to indicate failure for this combo
        return params, {}  # Corrected return type for failure


def optimize_strategy(
    strategy_short_name: str,
    config_path: str,
    symbol: str,
    data: pd.DataFrame,
    trade_units: float,
    optimization_metric: str = "cumulative_profit",
    commission_bps: float = 0.0,
    initial_balance: float = 10000.0,
    num_processes: int = DEFAULT_OPT_PROCESSES,  # Added processes arg
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Performs grid search optimization for a given strategy using parallel backtesting.

    Args:
        strategy_short_name: The short name of the strategy to optimize (e.g., "MACross").
        config_path: Path to the YAML configuration file.
        symbol: The trading symbol (e.g., "BTCUSDT") to load config for.
        data: Historical data DataFrame.
        trade_units: Units to trade.
        optimization_metric: The key in the backtest result to maximize.
        commission_bps: Commission fee in basis points passed to backtester.
        initial_balance: Starting balance for backtests.
        num_processes: Number of parallel processes to use for backtesting.

    Returns:
        A tuple containing: (best_params, best_result_dict, all_results_list) or (None, None, []) if error.
    """
    if strategy_short_name not in STRATEGY_MAP:
        logger.error(
            f"Unknown strategy name: {strategy_short_name}. Available: {list(STRATEGY_MAP.keys())}"
        )
        raise ValueError(
            f"Unknown strategy name: {strategy_short_name}. Available: {list(STRATEGY_MAP.keys())}"
        )

    strategy_class = STRATEGY_MAP[strategy_short_name]
    strategy_class_name = strategy_class.__name__

    try:
        # Load the specific grid for this symbol and strategy
        param_grid = load_param_grid_from_config(
            config_path, symbol, strategy_class_name
        )
    except (FileNotFoundError, ValueError, IOError) as e:
        logger.error(f"Stopping optimization due to error loading parameter grid: {e}")
        return None, None, []

    # --- Generate all parameter combinations ---
    param_combinations = list(
        generate_param_combinations(param_grid, strategy_class_name)
    )
    total_combinations = len(param_combinations)

    if total_combinations == 0:
        logger.warning(
            f"No valid parameter combinations generated for {strategy_class_name} on {symbol}. Check constraints."
        )
        return None, None, []

    logger.info(
        f"Generated {total_combinations} parameter combinations for {strategy_class_name} on {symbol}."
    )
    logger.info(f"Starting parallel backtesting with {num_processes} processes...")

    # --- Prepare arguments for the worker function ---
    worker_args_list = [
        {
            "params": params,
            "strategy_class": strategy_class,
            "symbol": symbol,
            "data": data,
            "trade_units": trade_units,
            "commission_bps": commission_bps,
            "initial_balance": initial_balance,
        }
        for params in param_combinations
    ]

    # --- Run backtests in parallel ---
    all_results_list = []
    best_result_so_far: Optional[Dict[str, Any]] = None
    best_params_so_far: Optional[Dict[str, Any]] = None
    completed_count = 0

    try:
        with Pool(processes=num_processes) as pool:
            # Using imap_unordered to get results as they complete (potentially better memory)
            # Each item in the iterator will be the tuple (params, result_dict)
            for params, result_dict in pool.imap_unordered(
                run_backtest_for_params, worker_args_list
            ):
                completed_count += 1
                if (
                    completed_count % PROGRESS_LOG_INTERVAL == 0
                    or completed_count == total_combinations
                ):
                    logger.info(
                        f"Processed {completed_count}/{total_combinations} combinations for {symbol}..."
                    )

                if not result_dict:  # Skip if backtest failed (returned empty dict)
                    continue

                # --- Track Best Result ---
                current_metric_value = result_dict.get(optimization_metric)

                if current_metric_value is None:
                    logger.warning(
                        f"Optimization metric '{optimization_metric}' not found in results for params: {params}. Skipping."
                    )
                    continue

                # Handle metrics where higher is better vs lower is better (e.g., max_drawdown)
                is_better = False
                if best_result_so_far is None:
                    is_better = True
                else:
                    best_metric_value = best_result_so_far.get(optimization_metric)
                    if (
                        best_metric_value is None
                    ):  # Should not happen if first result was valid
                        is_better = True
                    elif (
                        optimization_metric == "max_drawdown"
                    ):  # Lower is better for drawdown
                        # Ensure both are treated as positive for comparison
                        if abs(current_metric_value) < abs(best_metric_value):
                            is_better = True
                    elif (
                        current_metric_value > best_metric_value
                    ):  # Higher is better for others
                        is_better = True

                if is_better:
                    best_params_so_far = params
                    best_result_so_far = result_dict
                    # Log improvement
                    adjusted_params = adjust_params_for_printing(
                        params, strategy_class_name
                    )
                    logger.debug(
                        f"New best found: Metric({optimization_metric})={current_metric_value:.4f}, Params={adjusted_params}"
                    )

                # --- Store All Results (for saving details) ---
                # Combine params and results into a single dict for the list
                full_result_entry = {f"params.{k}": v for k, v in params.items()}
                full_result_entry.update(
                    {f"result_{k}": v for k, v in result_dict.items()}
                )
                all_results_list.append(full_result_entry)

    except KeyboardInterrupt:
        logger.warning(
            "Keyboard interrupt received during parallel backtesting. Terminating..."
        )
        # Pool context manager handles termination
        return (
            best_params_so_far,
            best_result_so_far,
            all_results_list,
        )  # Return what we have so far
    except Exception as e:
        logger.error(
            f"An error occurred during parallel processing: {e}", exc_info=True
        )
        return None, None, []  # Indicate failure

    if completed_count != total_combinations:
        logger.warning(
            f"Pool processing finished, but completed count ({completed_count}) does not match total combinations ({total_combinations}). Some jobs might have failed silently."
        )

    logger.info("Parallel backtesting finished.")

    # Log the final best result found
    if best_params_so_far and best_result_so_far:
        adjusted_best_params = adjust_params_for_printing(
            best_params_so_far, strategy_class_name
        )
        best_metric_final = best_result_so_far.get(optimization_metric)
        logger.info(
            f"Optimization complete. Best Result ({optimization_metric} = {best_metric_final:.4f}):"
        )
        logger.info(f"  Params: {adjusted_best_params}")
    else:
        logger.warning(
            "Optimization finished, but no valid best parameters were found."
        )

    return best_params_so_far, best_result_so_far, all_results_list


def adjust_params_for_printing(
    params: Dict[str, Any], strategy_class_name: str
) -> Dict[str, Any]:
    printable_params = params.copy()
    if strategy_class_name == "LongShortStrategy":
        if "return_thresh" in printable_params:
            rt = printable_params.pop("return_thresh")
            printable_params["return_thresh_low"] = rt[0]
            printable_params["return_thresh_high"] = rt[1]
        if "volume_thresh" in printable_params:
            vt = printable_params.pop("volume_thresh")
            printable_params["volume_thresh_low"] = vt[0]
            printable_params["volume_thresh_high"] = vt[1]
    # Add adjustments for other strategies if needed
    return printable_params


def save_best_params(
    output_config_path: str,
    symbol: str,
    strategy_class_name: str,
    best_params: Dict[str, Any],
):
    """Saves the best parameters found (incl. SL/TP/TSL) to the specified YAML output file."""
    if not best_params:
        logger.warning("No best parameters found to save.")
        return

    output_file = Path(output_config_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    full_results: Dict[str, Any] = {}
    if output_file.is_file():
        try:
            with open(output_file, "r") as f:
                full_results = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(
                f"Error reading existing results file {output_file}: {e}. Will overwrite."
            )
            full_results = {}

    # Adjust LongShortStrategy params back to low/high format before saving
    params_to_save = adjust_params_for_printing(best_params, strategy_class_name)

    # Update nested dictionary structure
    if symbol not in full_results:
        full_results[symbol] = {}
    full_results[symbol][strategy_class_name] = params_to_save

    try:
        with open(output_file, "w") as f:
            yaml.dump(full_results, f, indent=2, default_flow_style=False)
        logger.info(f"Best parameters saved to: {output_file}")
    except Exception as e:
        logger.error(
            f"Error writing best parameters to {output_file}: {e}", exc_info=True
        )


def main():
    parser = argparse.ArgumentParser(description="Optimize Strategy Parameters")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=STRATEGY_MAP.keys(),
        help="Strategy short name.",
    )
    parser.add_argument(
        "--symbol", required=True, help="Trading symbol (e.g., BTCUSDT)."
    )
    parser.add_argument(
        "--file", required=True, type=Path, help="Path to the historical data CSV file."
    )
    parser.add_argument(
        "--config",
        default="config/optimize_params.yaml",
        help="Path to the optimization parameters YAML file.",
    )
    parser.add_argument(
        "--output-config",
        default="config/best_params.yaml",
        help="Path to save the best parameters YAML file.",
    )
    parser.add_argument(
        "--opt-start",
        type=str,
        required=True,
        help="Optimization start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--opt-end", type=str, required=True, help="Optimization end date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--balance", type=float, default=10000.0, help="Initial backtest balance."
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Commission fee in basis points (e.g., 7.5 for 0.075%%).",
    )
    parser.add_argument(
        "--metric",
        default="cumulative_profit",
        help="Metric to optimize (from backtest results).",
    )
    parser.add_argument(
        "--units",
        type=float,
        default=1.0,
        help="Units per trade (e.g., 1 for 1 BTC, 0.1 for 0.1 ETH).",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save detailed results of all combinations to CSV.",
    )
    parser.add_argument(
        "--details-file",
        type=str,
        default=None,
        help="Optional filename for detailed results CSV (defaults to auto-generated).",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=DEFAULT_OPT_PROCESSES,
        help=f"Number of parallel processes for backtesting (default: {DEFAULT_OPT_PROCESSES}).",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )

    args = parser.parse_args()

    # Set logging level from args
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    try:
        logger.info(f"Starting optimization run with args: {args}")
        # Load FULL dataset first using the required --file argument
        full_data = load_csv_data(args.file, symbol=args.symbol)
        full_data.sort_index(inplace=True)
        logger.info(
            f"Loaded full dataset: {len(full_data)} rows from {full_data.index.min()} to {full_data.index.max()}"
        )

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
                    raise ValueError(
                        f"No data found within the specified optimization period: {args.opt_start} - {args.opt_end}"
                    )
            except Exception as e:
                logger.error(
                    f"Error slicing data for optimization period ({args.opt_start} - {args.opt_end}): {e}",
                    exc_info=True,
                )
                # Decide how to handle: exit or proceed with full data?
                # For safety, let's exit if slicing fails or results in empty data
                raise ValueError(f"Failed to slice data for optimization: {e}")

        logger.info(
            f"Using data slice for optimization: {len(opt_data)} rows ({slice_info})"
        )

        # Run Optimization on the potentially sliced data
        # Unpack all 3 returned values
        best_params, best_result_dict, all_results_list = optimize_strategy(
            strategy_short_name=args.strategy,
            config_path=args.config,
            symbol=args.symbol,
            data=opt_data,  # Pass the potentially sliced data
            trade_units=args.units,
            optimization_metric=args.metric,
            commission_bps=args.commission,
            initial_balance=args.balance,
            num_processes=args.processes,  # Pass num_processes
        )

        # --- Save Best Parameters ---
        if best_params:
            strategy_class = STRATEGY_MAP[args.strategy]
            strategy_class_name = strategy_class.__name__
            save_best_params(
                output_config_path=args.output_config,
                symbol=args.symbol,
                strategy_class_name=strategy_class_name,
                best_params=best_params,
            )
            logger.info(f"Best parameters saved to: {args.output_config}")
        else:
            logger.warning(
                "Optimization did not yield best parameters. Results file not updated."
            )

        if best_result_dict and isinstance(
            best_result_dict.get("performance_summary"), pd.DataFrame
        ):
            logger.info("\n--- Performance Summary for Best Parameters ---")
            summary_string = best_result_dict["performance_summary"].to_string()
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
                        row = res["params"].copy()  # Start with parameters
                        # Add other results, prefixing to avoid name collisions if needed
                        for k, v in res.items():
                            if (
                                k != "params"
                                and k != "performance_summary"
                                and k != "equity_curve"
                            ):  # Exclude complex objects
                                row[f"result_{k}"] = v
                        flat_results.append(row)

                    results_df = pd.DataFrame(flat_results)
                    results_df.to_csv(details_path, index=False)
                    logger.info(
                        f"Saved detailed optimization results to: {details_path}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error saving detailed optimization results: {e}",
                        exc_info=True,
                    )
            else:
                logger.warning(
                    "Detailed results requested, but no results were generated during optimization."
                )

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


if __name__ == "__main__":
    main()
