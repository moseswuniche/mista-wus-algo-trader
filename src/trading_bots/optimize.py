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
from tqdm import tqdm  # Added for progress bar
import sys
import inspect  # Added for inspecting strategy __init__ signatures

# Assuming strategies are accessible via this import path
from .strategies import (
    Strategy,
    LongShortStrategy,
    MovingAverageCrossoverStrategy,
    RsiMeanReversionStrategy,
    BollingerBandReversionStrategy,
)
from .backtest import run_backtest, calculate_sharpe_ratio, calculate_max_drawdown, calculate_profit_factor
from .technical_indicators import calculate_atr
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

# Add project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


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
            # Ensure atr_threshold_multiplier is None if intended
            if params.get("atr_threshold_multiplier") == "None":
                params["atr_threshold_multiplier"] = None
        elif strategy_class_name == "LongShortStrategy":
            # Pop volume threshold components
            # Assuming volume_z_thresh_low/high are the keys from the YAML
            vt_low = params.pop("volume_z_thresh_low")
            vt_high = params.pop("volume_z_thresh_high")

            # Add the expected volume_thresh tuple for the constructor
            params["volume_thresh"] = (vt_low, vt_high)

            # Remove the old fixed return threshold keys if they somehow persist
            params.pop("return_thresh_low", None)
            params.pop("return_thresh_high", None)

            # Ensure the new Z-score parameters are present (they should be loaded from YAML)
            # No need to pop/add them here, they should be passed directly
            # if 'return_z_score_period' not in params or 'return_z_score_threshold' not in params:
            #     logger.warning(f"Missing Z-score params for LongShortStrategy combo: {params}")
            #     continue # Or handle as error

            # Convert string 'None' to Python None for trend_filter_period if present
            if params.get("trend_filter_period") == "None":
                params["trend_filter_period"] = None

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
    """Worker function for parallel backtesting."""
    params = args_dict["params"]
    strategy_class = args_dict["strategy_class"]
    symbol = args_dict["symbol"]
    data = args_dict["data"]
    trade_units = args_dict["trade_units"]
    commission_bps = args_dict["commission_bps"]
    initial_balance = args_dict["initial_balance"]
    # TODO: Add filter params (apply_atr_filter, etc.) to args_dict if controlled globally
    apply_atr_filter = args_dict.get("apply_atr_filter", False)  # Example default
    allowed_trading_hours_utc = args_dict.get(
        "allowed_trading_hours_utc"
    )  # Example default
    apply_seasonality_to_symbols = args_dict.get(
        "apply_seasonality_to_symbols"
    )  # Example default

    # --- Get optional SL/TP values from params, default to None ---
    stop_loss_pct = params.get("stop_loss_pct")
    take_profit_pct = params.get("take_profit_pct")
    trailing_stop_loss_pct = params.get("trailing_stop_loss_pct")
    # --- Remove SL/TP/TSL from strategy params before instantiation ---
    strategy_params_from_grid = {
        k: v
        for k, v in params.items()
        if k not in ["stop_loss_pct", "take_profit_pct", "trailing_stop_loss_pct"]
    }

    # Ensure trend_filter_period is None, not the string "None"
    if strategy_params_from_grid.get("trend_filter_period") == "None":
        strategy_params_from_grid["trend_filter_period"] = None

    try:
        # --- Filter params to only those accepted by the strategy's __init__ ---
        init_signature = inspect.signature(strategy_class.__init__)
        valid_init_params = init_signature.parameters.keys()
        # Exclude 'self' if present (though **kwargs usually handles it)
        valid_init_params = {p for p in valid_init_params if p != 'self'}

        strategy_init_args = {
            k: v
            for k, v in strategy_params_from_grid.items()
            if k in valid_init_params
        }

        # Instantiate the strategy with the filtered parameter combination
        strategy_instance = strategy_class(**strategy_init_args)

        # Log start including symbol
        logger.info(f"Running backtest for: {strategy_class.__name__} on {symbol}")

        # Run the backtest with correct args
        result_dict = run_backtest(
            strategy=strategy_instance,
            symbol=args_dict["symbol"],
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
        # Return empty result on failure to avoid breaking the process
        return params, {}


def optimize_strategy(
    strategy_short_name: str,
    config_path: str,
    shared_worker_args: Dict,
    optimization_metric: str = "cumulative_profit",
    num_processes: int = DEFAULT_OPT_PROCESSES,  # Added processes arg
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Performs grid search optimization for a given strategy using parallel backtesting.

    Args:
        strategy_short_name: The short name of the strategy to optimize (e.g., "MACross").
        config_path: Path to the YAML configuration file.
        shared_worker_args: Dictionary containing shared arguments for backtesting.
        optimization_metric: The key in the backtest result to maximize.
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
            config_path, shared_worker_args["symbol"], strategy_class_name
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
            f"No valid parameter combinations generated for {strategy_class_name} on {shared_worker_args['symbol']}. Check constraints."
        )
        return None, None, []

    logger.info(
        f"Generated {total_combinations} parameter combinations for {strategy_class_name} on {shared_worker_args['symbol']}."
    )
    logger.info(f"Starting parallel backtesting with {num_processes} processes...")

    # --- Prepare arguments for the worker function ---
    worker_args_list = [
        {
            "params": params,
            "symbol": shared_worker_args["symbol"],
            "strategy_class": strategy_class,
            "data": shared_worker_args["data"],
            "trade_units": shared_worker_args["trade_units"],
            "commission_bps": shared_worker_args["commission_bps"],
            "initial_balance": shared_worker_args["initial_balance"],
            "apply_atr_filter": shared_worker_args["apply_atr_filter"],
            "atr_filter_period": shared_worker_args["atr_filter_period"],
            "atr_filter_multiplier": shared_worker_args["atr_filter_multiplier"],
            "atr_filter_sma_period": shared_worker_args["atr_filter_sma_period"],
            "apply_seasonality_filter": shared_worker_args["apply_seasonality_filter"],
            "allowed_trading_hours_utc": shared_worker_args[
                "allowed_trading_hours_utc"
            ],
            "apply_seasonality_to_symbols": shared_worker_args[
                "apply_seasonality_to_symbols"
            ],
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
                        f"Processed {completed_count}/{total_combinations} combinations for {shared_worker_args['symbol']}..."
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
    # --- Core Arguments ---
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

    # --- Backtest/Optimization Parameters ---
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
        default="sharpe_ratio",
        help="Metric to optimize (from backtest results).",
    )
    parser.add_argument(
        "--units",
        type=float,
        default=1.0,
        help="Units per trade (e.g., 1 for 1 BTC, 0.1 for 0.1 ETH). Fixed for optimization.",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",  # Handled by $(if ...) in Makefile
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
        help="Allowed trading hours in UTC (e.g., '5-17'). Required if seasonality filter is enabled.",
    )
    parser.add_argument(
        "--apply-seasonality-to-symbols",
        type=str,
        default="",
        help="Comma-separated list of symbols to apply seasonality filter to (if empty, applies to the main symbol).",
    )

    # --- Other Arguments ---
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

        # Validate filter arguments consistency
        if args.apply_seasonality_filter and not args.allowed_trading_hours_utc:
            raise ValueError(
                "--allowed-trading-hours-utc must be set when --apply-seasonality-filter is enabled."
            )

        # Load FULL dataset first using the required --file argument
        full_data = load_csv_data(str(args.file), symbol=args.symbol)
        full_data.sort_index(inplace=True)
        logger.info(
            f"Loaded full dataset: {len(full_data)} rows from {full_data.index.min()} to {full_data.index.max()}"
        )

        # --- Pre-calculate Indicators needed for Filters (e.g., ATR) ---
        # Doing this once on the full dataset slice used for optimization
        # Ensure data_utils has calculate_atr or similar
        if args.apply_atr_filter:
            full_data["atr"] = calculate_atr(full_data, period=args.atr_filter_period)
            if args.atr_filter_sma_period > 0:
                full_data["atr_sma"] = (
                    full_data["atr"].rolling(window=args.atr_filter_sma_period).mean()
                )
            logger.info(
                f"Pre-calculated ATR (period={args.atr_filter_period}, sma={args.atr_filter_sma_period}) for full dataset."
            )
        # --- End Indicator Pre-calculation ---

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

        # --- Prepare shared arguments for the worker function ---
        # Include filter parameters to be passed to run_backtest_for_params
        shared_worker_args = {
            "symbol": args.symbol,
            "data": opt_data,  # Pass the slice with pre-calculated indicators
            "trade_units": args.units,
            "commission_bps": args.commission,
            "initial_balance": args.balance,
            "apply_atr_filter": args.apply_atr_filter,
            "atr_filter_period": args.atr_filter_period,
            "atr_filter_multiplier": args.atr_filter_multiplier,
            "atr_filter_sma_period": args.atr_filter_sma_period,
            "apply_seasonality_filter": args.apply_seasonality_filter,
            "allowed_trading_hours_utc": args.allowed_trading_hours_utc,
            "apply_seasonality_to_symbols": args.apply_seasonality_to_symbols,
        }
        # --- End Shared Args Prep ---

        # Run Optimization on the potentially sliced data
        # This function now needs to accept shared_worker_args
        best_params, best_result_dict, all_results_list = optimize_strategy(
            strategy_short_name=args.strategy,
            config_path=args.config,
            shared_worker_args=shared_worker_args,  # Pass the dict
            optimization_metric=args.metric,
            num_processes=args.processes,
        )

        # --- Save Best Parameters ---
        if best_params:
            strategy_class = STRATEGY_MAP[args.strategy]
            strategy_class_name = strategy_class.__name__
            # Adjust params before saving if needed (like removing filter args if they are in best_params)
            # Or ensure save_best_params only saves strategy-specific ones
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

        # --- Save Detailed Results if requested ---
        if args.save_details:
            if all_results_list:
                try:
                    # Define default filename if not provided
                    if args.details_file is None:
                        details_dir = Path("results/optimization")
                        details_dir.mkdir(parents=True, exist_ok=True)
                        filename_ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                        base_name = f"{args.strategy}_{args.symbol}"

                        # Add filter info to filename if applied
                        filter_suffix = ""
                        if args.apply_atr_filter:
                            filter_suffix += f"_atr{args.atr_filter_period}x{args.atr_filter_multiplier:.1f}sma{args.atr_filter_sma_period}"
                        if args.apply_seasonality_filter:
                            filter_suffix += f"_season{args.allowed_trading_hours_utc.replace('-','')}"
                            if args.apply_seasonality_to_symbols:
                                # Keep symbols short in filename
                                syms = args.apply_seasonality_to_symbols.split(",")
                                filter_suffix += f"_{''.join(s[:3] for s in syms)}"

                        # Sanitize suffix
                        filter_suffix = sanitize_filename(filter_suffix)

                        filename = f"{base_name}{filter_suffix}_optimize_details_{filename_ts}.csv"
                        details_path = details_dir / filename
                    else:
                        details_path = Path(args.details_file)
                        details_path.parent.mkdir(parents=True, exist_ok=True)

                    # Flatten the results for CSV
                    flat_results = []
                    if all_results_list:
                        # Assuming the first item has all keys needed for columns
                        # Get param keys by splitting 'params.'
                        param_keys = sorted(
                            [
                                k
                                for k in all_results_list[0].keys()
                                if k.startswith("params.")
                            ]
                        )
                        result_keys = sorted(
                            [
                                k
                                for k in all_results_list[0].keys()
                                if k.startswith("result_")
                            ]
                        )
                        # Add non-prefixed keys if any (shouldn't be any with current structure)
                        other_keys = sorted(
                            [
                                k
                                for k in all_results_list[0].keys()
                                if not k.startswith(("params.", "result_"))
                            ]
                        )

                        # Ensure consistent column order
                        all_columns = param_keys + result_keys + other_keys

                        for res in all_results_list:
                            # Ensure all keys exist in each dict, adding None if missing
                            flat_results.append(
                                {col: res.get(col) for col in all_columns}
                            )

                    details_df = pd.DataFrame(flat_results)
                    # Optional: Sort by the optimization metric descending (higher is better)
                    # Handle max_drawdown separately (lower is better)
                    metric_col = f"result_{args.metric}"
                    if metric_col in details_df.columns:
                        ascending_sort = args.metric == "max_drawdown"
                        details_df.sort_values(
                            by=metric_col, ascending=ascending_sort, inplace=True
                        )

                    details_df.to_csv(details_path, index=False)
                    logger.info(
                        f"Detailed optimization results saved to: {details_path}"
                    )
                except Exception as e:
                    logger.error(f"Failed to save detailed results: {e}", exc_info=True)
            else:
                logger.warning(
                    "--save-details was specified, but no results were generated."
                )

    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
    except ValueError as e:
        logger.error(f"Configuration or data error: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during optimization: {e}", exc_info=True
        )
    finally:
        logger.info("--- Optimization script finished. ---")


if __name__ == "__main__":
    main()

# --- Walk-Forward Analysis Structure (Conceptual Outline) ---
# def run_walk_forward_optimization(
#     strategy_short_name: str,
#     config_path: str,
#     symbol: str,
#     full_data: pd.DataFrame, # Pass the complete dataset
#     train_split_ratio: float = 0.6, # e.g., 60% for training
#     num_windows: int = 5, # Number of walk-forward windows
#     optimization_metric: str = "cumulative_profit",
#     # ... other params like trade_units, commission, initial_balance, num_processes ...
# ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
#     """Performs walk-forward optimization.
#        Outline:
#        1. Split full_data into num_windows train/test sets (rolling or anchored).
#        2. Loop through each window:
#           a. Find best params on train set using optimize_strategy (or similar logic).
#           b. Run backtest with BEST train params on the corresponding test set.
#           c. Store test set results (metrics, trades).
#        3. Aggregate and analyze results from all test sets.
#        4. Return aggregated results and potentially the best params sequence.
#     """
#     logger.info(f"Starting Walk-Forward Optimization for {symbol} / {strategy_short_name}")
#     # --- Placeholder for data splitting logic --- #
#     # window_size = len(full_data) // num_windows
#     # test_size = int(window_size * (1.0 - train_split_ratio))
#     # train_size = window_size - test_size
#     all_test_results = []
#     best_params_sequence = []
#
#     # for i in range(num_windows): # Or calculate start/end indices
#         # train_data = ... # Slice full_data
#         # test_data = ... # Slice full_data
#
#         # logger.info(f"Walk-Forward Window {i+1}/{num_windows}: Optimizing on Train Data...")
#         # best_train_params, _, _ = optimize_strategy(
#         #     strategy_short_name=strategy_short_name,
#         #     config_path=config_path,
#         #     symbol=symbol,
#         #     data=train_data,
#         #     trade_units=trade_units,
#         #     optimization_metric=optimization_metric,
#         #     # ... pass other necessary args ...
#         # )
#
#         # if best_train_params:
#         #     logger.info(f"Window {i+1}: Best Train Params Found: {best_train_params}")
#         #     best_params_sequence.append(best_train_params)
#
#         #     logger.info(f"Window {i+1}: Running Backtest on Test Data with Best Train Params...")
#         #     strategy_class = STRATEGY_MAP[strategy_short_name]
#         #     strategy_instance = strategy_class(**best_train_params) # Assuming best_params is ready for init
#         #     test_result = run_backtest(
#         #         data=test_data,
#         #         strategy=strategy_instance,
#         #         symbol=symbol,
#         #         units=trade_units,
#         #         # ... pass other necessary args including SL/TP/TSL from best_train_params ...
#         #     )
#         #     all_test_results.append({"window": i+1, "params": best_train_params, "results": test_result})
#         # else:
#         #     logger.warning(f"Window {i+1}: No best parameters found during training optimization.")
#
#     # --- Aggregate Results --- #
#     # Combine performance summaries, recalculate overall metrics, etc.
#     logger.info("Walk-Forward Optimization Complete. Aggregating results...")
#     # aggregated_results = ...
#     # return aggregated_results, best_params_sequence
#     return None, None, [] # Placeholder return
