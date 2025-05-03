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
import sys
import inspect  # Added for inspecting strategy __init__ signatures
from functools import partial  # Added for partial function application
import json  # Added for stable serialization of params
from logging.handlers import (
    QueueHandler,
    QueueListener,
)  # Added for multiprocessing logging
import multiprocessing  # Added for multiprocessing
import logging.handlers
import time
import traceback  # For listener error reporting
import re  # Import re for sanitize_filename if not already imported
from tqdm import tqdm  # Import tqdm

# --- Import Pydantic Models --- << NEW >>
from .config_models import OptimizeParamsConfig, ValidationError
from .config_models import BacktestRunConfig

# Assuming strategies are accessible via this import path
from .strategies import (
    Strategy,
    LongShortStrategy,
    MovingAverageCrossoverStrategy,
    RsiMeanReversionStrategy,
    BollingerBandReversionStrategy,
)
from .backtest import (
    run_backtest,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
)
from .technical_indicators import calculate_atr
from .data_utils import load_csv_data

# Re-add module-level logger instance
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

# --- Multiprocessing Logging Setup --- Corrected ---

# <<< Worker Globals >>>
worker_log_queue: Optional[multiprocessing.Queue] = None
worker_data: Optional[pd.DataFrame] = None
worker_shared_args: Dict[str, Any] = {}
WORKER_STRATEGY_MAP: Dict[str, type] = {}  # Map populated in initializer
# <<< End Worker Globals >>>

# --- Remove Parquet Globals and Cleanup Function ---
# (Removed _parquet_writer_instance, _details_filepath_global, _atexit_registered)
# (Removed _cleanup_parquet_writer function)
# --- End Globals and Cleanup ---


def worker_log_configurer(log_queue: multiprocessing.Queue):
    """Configures logging for a worker process to send ONLY to the queue."""
    root = logging.getLogger()
    # Ensure root logger exists
    if not root:
        print(
            f"Worker {multiprocessing.current_process().pid}: Root logger not found!",
            file=sys.stderr,
        )
        return

    # *** IMPORTANT: Remove existing handlers from worker's root logger ***
    if root.hasHandlers():
        print(
            f"Worker {multiprocessing.current_process().pid}: Removing {len(root.handlers)} existing handler(s) from root logger.",
            file=sys.stderr,
        )
        root.handlers.clear()

    # Add ONLY the QueueHandler
    queue_handler = QueueHandler(log_queue)
    root.addHandler(queue_handler)
    # Set the level for this specific handler - messages BELOW this level won't be put on the queue
    # The overall root level will be set by the main process setup.
    root.setLevel(logging.DEBUG)  # Send DEBUG and higher from workers to the queue


def pool_worker_initializer_with_data(
    log_q: multiprocessing.Queue, shared_worker_args_for_init: Dict
):
    """Initializer for worker processes.
    Sets up logging and stores shared data in global variables.
    """
    global worker_log_queue, worker_data, worker_shared_args, WORKER_STRATEGY_MAP

    print(
        f"Initializing worker {multiprocessing.current_process().pid} with queue {id(log_q)}...",
        file=sys.stderr,  # Basic print for debug
    )

    # 1. Configure Logging
    worker_log_queue = log_q
    worker_log_configurer(worker_log_queue)

    # 2. Store Shared Data/Args
    # Make a copy to ensure each worker has its own reference (though DataFrame might still share memory)
    worker_shared_args = shared_worker_args_for_init.copy()
    worker_data = worker_shared_args.pop(
        "data", None
    )  # Extract data to separate global
    if not isinstance(worker_data, pd.DataFrame) or worker_data.empty:
        print(
            f"CRITICAL ERROR in worker {multiprocessing.current_process().pid}: Did not receive valid data in initializer! Type: {type(worker_data)}",
            file=sys.stderr,
        )
        # Maybe raise an error or log to stderr? Pool might handle worker exit.
        logger = logging.getLogger("WorkerInitError")
        logger.error("Worker did not receive data DataFrame during initialization.")
        # Attempting to log to queue if possible
        if worker_log_queue:
            try:
                err_record = logger.makeRecord(
                    name="WorkerInitError",
                    level=logging.CRITICAL,
                    fn="",
                    lno=0,
                    msg="Worker data is None",
                    args=(),
                    exc_info=None,
                    func="pool_worker_initializer_with_data",
                )
                worker_log_queue.put(err_record)
            except Exception as e:
                print(
                    f"WorkerInitError: Failed to put log record in queue: {e}",
                    file=sys.stderr,
                )
        # Exiting worker might be necessary if data is crucial
        # sys.exit(1) # Or let the pool handle it

    # 3. Populate Worker Strategy Map (avoids repeated imports in worker function)
    try:
        from .strategies import (
            LongShortStrategy,
            MovingAverageCrossoverStrategy,
            RsiMeanReversionStrategy,
            BollingerBandReversionStrategy,
        )

        WORKER_STRATEGY_MAP = {
            "LongShort": LongShortStrategy,
            "MACross": MovingAverageCrossoverStrategy,
            "RSIReversion": RsiMeanReversionStrategy,
            "BBReversion": BollingerBandReversionStrategy,
        }
    except ImportError as e:
        print(
            f"CRITICAL ERROR in worker {multiprocessing.current_process().pid}: Could not import strategies: {e}"
        )
        # Log/Handle error similar to data error
        logger = logging.getLogger("WorkerInitError")
        logger.critical(f"Worker could not import strategies: {e}")
        if worker_log_queue:
            try:
                err_record = logger.makeRecord(
                    name="WorkerInitError",
                    level=logging.CRITICAL,
                    fn="",
                    lno=0,
                    msg=f"Strategy import failed: {e}",
                    args=(),
                    exc_info=None,
                    func="pool_worker_initializer_with_data",
                )
                worker_log_queue.put(err_record)
            except Exception as log_e:
                print(
                    f"WorkerInitError: Failed to put log record in queue: {log_e}",
                    file=sys.stderr,
                )
        # sys.exit(1)

    print(
        f"Worker {multiprocessing.current_process().pid} initialized successfully. Data shape: {worker_data.shape if worker_data is not None else 'None'}",
        file=sys.stderr,
    )


# <<< START MOVED HELPER FUNCTIONS >>>

# --- Parameter Loading & Processing ---
# --- (Moved Here - BEFORE optimize_strategy) ---


# <<< Helper function to create stable hashable tuple representation of params >>>
def params_to_tuple_rep(params: Dict[str, Any], precision: int = 8) -> tuple:
    """Creates a stable, hashable tuple representation of a parameter dictionary,
       normalizing numeric types, sorting keys, and handling complex types.
    Args:
        params: The parameter dictionary.
        precision: The number of decimal places to round floats to.
    Returns:
        A hashable tuple representation.
    """
    items: List[Tuple[str, Any]] = []  # Explicitly type items list
    logger = logging.getLogger(__name__)  # Use logger defined in the module
    for k, v in sorted(params.items()):
        # Handle floats specifically for rounding
        if isinstance(v, float):
            try:
                rounded_v = round(v, precision)
                if rounded_v == -0.0:
                    rounded_v = 0.0
                items.append((k, rounded_v))
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not round float parameter '{k}' value {repr(v)}: {e}. Using original value."
                )
                items.append((k, v))
        # Keep ints as ints
        elif isinstance(v, int):
            items.append((k, v))
        elif isinstance(v, tuple):
            # Round float elements within tuples
            try:
                rounded_tuple_elements = []
                for elem in v:
                    if isinstance(elem, float):
                        rounded_elem = round(elem, precision)
                        if rounded_elem == -0.0:
                            rounded_elem = 0.0
                        rounded_tuple_elements.append(rounded_elem)
                    else:
                        rounded_tuple_elements.append(elem)  # Keep non-floats as is
                items.append((k, tuple(rounded_tuple_elements)))
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not round float elements in tuple parameter '{k}' value {repr(v)}: {e}. Using original value."
                )
                items.append((k, v))  # Use original tuple on error
        elif isinstance(v, list):
            # Convert lists to tuples for hashability, round floats within
            try:
                rounded_list_elements = []
                for elem in v:
                    if isinstance(elem, float):
                        rounded_elem = round(elem, precision)
                        if rounded_elem == -0.0:
                            rounded_elem = 0.0
                        rounded_list_elements.append(rounded_elem)
                    else:
                        rounded_list_elements.append(elem)
                # Correctly indented items.append block inside the try block
                items.append(
                    (
                        k,
                        tuple(
                            sorted(
                                rounded_list_elements,
                                key=lambda x: (
                                    (isinstance(x, type(None)), x)
                                    if isinstance(
                                        x, (int, float, str, bool, type(None))
                                    )
                                    else (2, type(x).__name__)
                                ),
                            )
                        ),
                    )
                )  # Sort list elements for stability
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not process list parameter '{k}' value {repr(v)}: {e}. Converting to tuple directly."
                )
                items.append((k, tuple(v)))  # Basic tuple conversion on error
        else:
            # For other hashable types (bool, str, None), append directly
            items.append((k, v))

    return tuple(items)


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """Removes or replaces characters invalid for filenames."""
    # Allow alphanumeric, underscore, hyphen, period
    valid_chars = set(string.ascii_letters + string.digits + "_-.")
    # Replace invalid characters with the replacement character
    sanitized = "".join(c if c in valid_chars else replacement for c in filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Remove consecutive replacements (e.g., ___)
    sanitized = re.sub(f"{re.escape(replacement)}+", replacement, sanitized)
    # Remove leading/trailing replacements
    sanitized = sanitized.strip(replacement)
    # Limit length (optional, but good practice)
    max_len = 200
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]
    return sanitized


# <<< END MOVED HELPER FUNCTIONS >>>


# --- Worker function for parallel backtesting ---
# *** Modified signature: accepts ONLY params dict ***
# *** Uses global variables set by initializer ***
def run_backtest_for_params(
    params: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    # Get logger configured by initializer
    logger = logging.getLogger(__name__)

    # Access globals
    global worker_data, worker_shared_args, WORKER_STRATEGY_MAP

    # +++ ADD DEBUG LOGGING FOR PARAMS ID +++
    logger.debug(
        f"Worker {multiprocessing.current_process().pid} ENTRY - Params object ID: {id(params)}"
    )
    # +++ END DEBUG LOGGING +++

    # Check if globals were initialized correctly
    if worker_data is None or not worker_shared_args or not WORKER_STRATEGY_MAP:
        logger.error(
            f"Worker {multiprocessing.current_process().pid} found uninitialized globals! Skipping task."
        )
        return None, None

    # --- Extract Base Config from Shared Args --- << MODIFIED >>
    try:
        strategy_short_name = worker_shared_args["strategy_short_name"]

        # Create the BacktestRunConfig object
        # Start with shared/fixed values
        run_config_data = {
            "symbol": worker_shared_args["symbol"],
            "initial_balance": worker_shared_args.get("initial_balance", 10000.0),
            "commission_bps": worker_shared_args.get("commission_bps", 0.0),
            "units": worker_shared_args["units"],
            "strategy_short_name": strategy_short_name,
            # Global filter flags (these determine if filters are *ever* applied)
            "apply_atr_filter": worker_shared_args.get("apply_atr_filter", False),
            "apply_seasonality_filter": worker_shared_args.get(
                "apply_seasonality_filter", False
            ),
            # Default filter parameters (used if the grid doesn't override)
            "atr_filter_period": worker_shared_args.get("atr_filter_period", 14),
            "atr_filter_multiplier": worker_shared_args.get(
                "atr_filter_multiplier", 1.5
            ),
            "atr_filter_sma_period": worker_shared_args.get(
                "atr_filter_sma_period", 100
            ),
            "allowed_trading_hours_utc": worker_shared_args.get(
                "allowed_trading_hours_utc"
            ),
            "apply_seasonality_to_symbols": worker_shared_args.get(
                "apply_seasonality_to_symbols"
            ),
        }

        # --- Merge Grid Parameters into Config --- #
        # Strategy-specific parameters
        # Filter out non-strategy params before assigning
        strategy_class = WORKER_STRATEGY_MAP.get(strategy_short_name)
        if not strategy_class:
            logger.error(f"Strategy {strategy_short_name} not found in worker map.")
            return params, None  # Return params for identification
        sig = inspect.signature(strategy_class.__init__)  # type: ignore[misc]
        valid_init_params = {p for p in sig.parameters if p != "self"}
        run_config_data["strategy_params"] = {
            k: v for k, v in params.items() if k in valid_init_params
        }

        # Risk Management parameters from grid
        run_config_data["stop_loss_pct"] = params.get("stop_loss_pct")
        run_config_data["take_profit_pct"] = params.get("take_profit_pct")
        run_config_data["trailing_stop_loss_pct"] = params.get("trailing_stop_loss_pct")

        # Filter parameters *from the grid* that might override defaults
        # Check if grid explicitly sets apply flags (if they were varied)
        if "apply_atr_filter" in params:
            run_config_data["apply_atr_filter"] = params["apply_atr_filter"]
        if (
            "apply_seasonality" in params
        ):  # Use 'apply_seasonality' as key from YAML grid
            run_config_data["apply_seasonality_filter"] = params["apply_seasonality"]

        # Update filter values if present in grid *and corresponding apply flag is true*
        if run_config_data["apply_atr_filter"]:
            if "atr_filter_period" in params:
                run_config_data["atr_filter_period"] = params["atr_filter_period"]
            if "atr_filter_threshold" in params:  # Key from YAML
                run_config_data["atr_filter_multiplier"] = params[
                    "atr_filter_threshold"
                ]
            if "atr_filter_sma_period" in params:
                run_config_data["atr_filter_sma_period"] = params[
                    "atr_filter_sma_period"
                ]

        if run_config_data["apply_seasonality_filter"]:
            # Construct hours string from grid params if available
            start_hour = params.get("seasonality_start_hour")
            end_hour = params.get("seasonality_end_hour")
            if start_hour is not None and end_hour is not None:
                run_config_data["allowed_trading_hours_utc"] = (
                    f"{start_hour}-{end_hour}"
                )
            # Override symbols list if present in grid
            if "apply_seasonality_to_symbols" in params:
                run_config_data["apply_seasonality_to_symbols"] = params[
                    "apply_seasonality_to_symbols"
                ]

        # --- Validate and Create Config Object --- #
        try:
            backtest_config = BacktestRunConfig(**run_config_data)
        except ValidationError as e:
            logger.error(
                f"Failed to validate BacktestRunConfig for params {params}:\n{e}"
            )
            return params, None

    except Exception as e:
        logger.error(
            f"Error preparing BacktestRunConfig in worker for params {params}: {e}",
            exc_info=True,
        )
        return params, None

    # --- Call Refactored run_backtest --- << MODIFIED >>
    try:
        result_metrics = run_backtest(
            data=worker_data,  # Use global data
            config=backtest_config,  # Pass the config object
        )

        if not result_metrics:
            logger.warning(
                f"Worker {multiprocessing.current_process().pid}: run_backtest function returned None or empty metrics for params: {params}"
            )

        # --- Remove detailed trade log BEFORE returning from worker ---
        if result_metrics and "result_performance_summary" in result_metrics:
            del result_metrics["result_performance_summary"]
            logger.debug("Removed result_performance_summary from worker result.")
        # --- End removal ---

        # Return the original params (passed into function) and the results
        return params, result_metrics

    except Exception as e:
        logger.error(
            f"Exception during backtest run for params {params}: {e}", exc_info=True
        )
        return params, None  # Return params even on failure, but None for results


# --- Helper Function: Load and Prepare Parameter Grid --- << NEW >>
def _load_and_prepare_param_grid(
    full_config: Dict,  # Expect validated config from main
    symbol: str,
    strategy_class_name: str,
    cmd_apply_atr_filter: bool,
    cmd_apply_seasonality_filter: bool,
) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
    """Loads grid for symbol/strategy, pre-filters based on cmd flags, normalizes.
    Returns:
        Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
            - Processed parameter grid ready for combination generation.
            - Original unprocessed grid for type reference.
    Raises:
        ValueError: If symbol/strategy not found or grid format is invalid.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Preparing param grid for {symbol}/{strategy_class_name}")

    if symbol not in full_config:
        raise ValueError(f"Symbol '{symbol}' not found in validated config.")

    symbol_config = full_config[symbol]
    if strategy_class_name not in symbol_config:
        raise ValueError(
            f"Strategy '{strategy_class_name}' not found for symbol '{symbol}' in config."
        )

    loaded_grid_raw = symbol_config[strategy_class_name]
    # Keep a copy of the raw grid for type inference later
    source_param_grid_for_types = loaded_grid_raw.copy()

    # --- Pre-filter Grid based on Command-Line Flags --- #
    loaded_grid = loaded_grid_raw.copy()  # Work on a copy
    original_grid_keys = set(loaded_grid.keys())

    if not cmd_apply_atr_filter:
        atr_params_to_remove = [
            "apply_atr_filter",
            "atr_filter_period",
            "atr_filter_multiplier",
            "atr_filter_threshold",
            "atr_filter_sma_period",
        ]
        removed_atr_count = 0
        for param in atr_params_to_remove:
            if loaded_grid.pop(param, None) is not None:
                removed_atr_count += 1
        if removed_atr_count > 0:
            logger.info(
                f"CMD Flag --apply-atr-filter=OFF: Removed {removed_atr_count} ATR params."
            )

    if not cmd_apply_seasonality_filter:
        seasonality_params_to_remove = [
            "apply_seasonality_filter",
            "apply_seasonality",
            "allowed_trading_hours_utc",
            "apply_seasonality_to_symbols",
            "seasonality_start_hour",
            "seasonality_end_hour",
        ]
        removed_season_count = 0
        for param in seasonality_params_to_remove:
            if loaded_grid.pop(param, None) is not None:
                removed_season_count += 1
        if removed_season_count > 0:
            logger.info(
                f"CMD Flag --apply-seasonality-filter=OFF: Removed {removed_season_count} seasonality params."
            )
    # --- End Pre-filter Grid ---

    # --- Process Remaining Parameter Grid (Normalize, Dedupe, Sort) --- #
    param_grid: Dict[str, List[Any]] = {}
    if not loaded_grid:
        logger.warning(
            f"Parameter grid for {symbol}/{strategy_class_name} is empty after command-line filtering."
        )
        # Return empty grid and the original source grid
        return {}, source_param_grid_for_types

    for key, value in loaded_grid.items():
        # Pydantic validation in main should ensure value is a list
        # Add basic check just in case
        if not isinstance(value, list):
            logger.error(
                f"Unexpected non-list value for parameter '{key}' in {symbol}/{strategy_class_name} after validation. Value: {value}"
            )
            # Or raise error? For now, skip this key
            continue

        processed_values_dict = {}
        for item in value:
            processed_item = item
            key_for_dedupe: Any = item  # Type hint for clarity
            # Normalize 'none' string
            if isinstance(item, str) and item.lower() == "none":
                processed_item = None
                key_for_dedupe = None
            # Use float for numeric keys to handle int/float duplicates (e.g., 1 vs 1.0)
            elif isinstance(item, (int, float)):
                key_for_dedupe = float(item)

            if key_for_dedupe not in processed_values_dict:
                processed_values_dict[key_for_dedupe] = processed_item

        unique_values_list = list(processed_values_dict.values())

        try:
            # Define sorting key function locally
            def sort_key(x):
                if x is None:
                    return (0, None)
                if isinstance(x, bool):
                    return (1, x)
                if isinstance(x, (int, float)):
                    return (2, x)
                if isinstance(x, str):
                    return (3, x)
                return (4, str(x))  # Fallback for other types

            sorted_unique_values = sorted(unique_values_list, key=sort_key)
            param_grid[key] = sorted_unique_values
            logger.debug(
                f"Processed param '{key}': Original={value}, UniqueSorted={sorted_unique_values}"
            )
        except TypeError as e:
            logger.warning(
                f"Sorting failed for parameter '{key}' values ({unique_values_list}): {e}. Using original order of unique items."
            )
            param_grid[key] = unique_values_list
    # --- End Process Remaining Parameter Grid --- #

    return param_grid, source_param_grid_for_types


# --- End Helper Function ---


# --- Helper Function: Generate Parameter Combinations --- << NEW >>
def _generate_combinations(
    param_grid: Dict[str, List[Any]],
    strategy_class_name: str,
    completed_param_reps: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Generates unique parameter combinations, handling flags/dependencies and resuming.
    Args:
        param_grid: The processed parameter grid (output from _load_and_prepare_param_grid).
        strategy_class_name: Name of the strategy (for potential special handling).
        completed_param_reps: Set of tuple representations of previously completed params.
    Returns:
        List of unique parameter dictionaries to run backtests for.
    """
    logger = logging.getLogger(__name__)
    completed_param_reps = completed_param_reps or set()

    if not param_grid:
        logger.warning("Parameter grid is empty, no combinations to generate.")
        return []

    # --- Define Flags and Dependencies (Copied from optimize_strategy) ---
    # This logic depends on the *processed* param_grid keys
    flags_and_deps = {
        "apply_seasonality": [
            "seasonality_start_hour",
            "seasonality_end_hour",
        ],
        # Add other flags defined *in the YAML* here if needed
    }

    # Separate grid into flags, dependents, and independents
    flag_grid = {k: param_grid.pop(k) for k in flags_and_deps if k in param_grid}
    dependent_params = {
        dep: param_grid.pop(dep)
        for flag, deps in flags_and_deps.items()
        for dep in deps
        if dep in param_grid
    }
    independent_params = param_grid  # Remaining params

    # --- Helper Function for Smart Combination Generation (Copied from optimize_strategy) ---
    def generate_smart_combinations(
        flag_grid, dependent_params, independent_params, flags_and_deps
    ):
        independent_names = list(independent_params.keys())
        independent_values = list(independent_params.values())
        independent_combinations = list(itertools.product(*independent_values))

        flag_names = list(flag_grid.keys())
        flag_values = list(flag_grid.values())
        flag_combinations = list(itertools.product(*flag_values))

        total_combinations = 0
        for flag_combo_values in flag_combinations:
            flag_combo_dict = dict(zip(flag_names, flag_combo_values))

            active_dependent_params = {}
            fixed_dependent_params = {}
            for flag_name, is_active in flag_combo_dict.items():
                deps = flags_and_deps.get(flag_name, [])
                for dep_name in deps:
                    if dep_name in dependent_params:
                        if is_active:
                            active_dependent_params[dep_name] = dependent_params[
                                dep_name
                            ]
                        else:
                            if dependent_params[dep_name]:  # Ensure list is not empty
                                fixed_dependent_params[dep_name] = dependent_params[
                                    dep_name
                                ][0]
                            else:
                                fixed_dependent_params[dep_name] = (
                                    None  # Or handle empty list case
                                )

            active_dep_names = list(active_dependent_params.keys())
            active_dep_values = list(active_dependent_params.values())

            dependent_combinations = (
                itertools.product(*active_dep_values)
                if active_dependent_params
                else [()]
            )

            for dep_combo_values in dependent_combinations:
                active_dep_combo_dict = dict(zip(active_dep_names, dep_combo_values))

                for indep_combo_values in independent_combinations:
                    indep_combo_dict = dict(zip(independent_names, indep_combo_values))
                    final_params = {
                        **flag_combo_dict,
                        **fixed_dependent_params,
                        **active_dep_combo_dict,
                        **indep_combo_dict,
                    }
                    total_combinations += 1
                    yield final_params
        # Removed logging total count from inner generator

    # --- End Inner Smart Combination Generator ---

    # --- Generate Unique Parameter Combinations Using Smart Generator --- #
    unique_rep_to_params_map: Dict[tuple, Dict[str, Any]] = {}
    smart_combination_generator = generate_smart_combinations(
        flag_grid, dependent_params, independent_params, flags_and_deps
    )
    generated_count = 0

    logger.info("Generating and deduplicating unique parameter combinations...")
    for i, params in enumerate(smart_combination_generator):
        generated_count += 1

        # Special handling for LongShortStrategy tuple params (if needed)
        if strategy_class_name == "LongShortStrategy":
            rt_low = params.pop("return_thresh_low", None)
            rt_high = params.pop("return_thresh_high", None)
            vt_low = params.pop("volume_thresh_low", None)
            vt_high = params.pop("volume_thresh_high", None)
            if rt_low is not None and rt_high is not None:
                if "return_thresh" not in params:
                    params["return_thresh"] = (rt_low, rt_high)
            if vt_low is not None and vt_high is not None:
                if "volume_thresh" not in params:
                    params["volume_thresh"] = (vt_low, vt_high)

        rep = params_to_tuple_rep(params, precision=8)

        if i < 5:
            logger.debug(f"Generated Param Set [{i}]: Params={repr(params)}, Rep={rep}")
        log_interval = 500
        if generated_count % log_interval == 0:
            logger.debug(f"Generated {generated_count} combinations so far...")

        if rep in unique_rep_to_params_map:
            continue
        unique_rep_to_params_map[rep] = params

    total_unique_combinations = len(unique_rep_to_params_map)
    logger.info(f"Smart generator yielded {generated_count} combinations.")
    logger.info(
        f"Deduplication complete: {total_unique_combinations} unique combinations identified initially."
    )
    duplicate_count = generated_count - total_unique_combinations
    if duplicate_count > 0:
        logger.info(f"  ({duplicate_count} duplicate combinations ignored)")

    # --- Filter based on completed_param_reps (Resume) --- #
    unique_params_list = list(unique_rep_to_params_map.values())
    if not completed_param_reps:
        logger.info(f"Running all {total_unique_combinations} unique combinations.")
        return unique_params_list
    else:
        params_to_run = []
        skipped_count = 0
        logger.debug(
            f"--- Resuming: Comparing {total_unique_combinations} generated params against {len(completed_param_reps)} loaded reps ---"
        )
        for params in unique_params_list:
            rep = params_to_tuple_rep(params)
            if rep in completed_param_reps:
                skipped_count += 1
            else:
                params_to_run.append(params)
        logger.debug(f"--- Resume Check Finished: Total Skipped={skipped_count} ---")

        remaining_count = len(params_to_run)
        logger.info(f"Resume: Skipped {skipped_count} already completed parameters.")
        logger.info(f"Resume: {remaining_count} parameters remaining to run.")
        return params_to_run


# --- End Helper Function ---


# --- Helper Function: Run Backtests in Parallel --- << NEW >>
def _run_parallel_backtests(
    unique_params_for_backtest: List[Dict[str, Any]],
    num_processes: int,
    log_queue: Optional[multiprocessing.Queue],
    shared_worker_args: Dict,
    save_details: bool,
    details_filepath: Optional[Path],
    optimization_metric: str,
    strategy_short_name: str,  # For logging/progress bar
    symbol: str,  # For logging/progress bar
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Runs backtests for the given param combinations in parallel using Pool.imap_unordered.

    Handles progress bar (tqdm), incremental CSV saving (if enabled),
    and finds the best parameters/metrics based *only on the results from this run*.

    Returns:
        Tuple containing the best parameters dictionary and best metrics dictionary
        found during this specific parallel execution run.
    """
    logger = logging.getLogger(__name__)
    unique_combinations_to_run_count = len(unique_params_for_backtest)

    if unique_combinations_to_run_count == 0:
        logger.warning(
            "No parameter combinations provided to _run_parallel_backtests. Skipping."
        )
        return None, None

    logger.info(
        f"Starting parallel backtesting for {unique_combinations_to_run_count} unique combinations using {num_processes} processes..."
    )

    # Determine if metric should be minimized
    minimize_metric = optimization_metric in ["max_drawdown"]

    # Initialize tracking for the best result *within this run*
    run_best_metric_value = float("inf") if minimize_metric else float("-inf")
    run_best_params = None
    run_best_metrics_dict = None

    # Prepare initializer arguments for the pool
    init_args_for_worker = shared_worker_args.copy()
    init_args_for_worker["strategy_short_name"] = strategy_short_name
    pool_initializer_func = None
    initializer_args: Optional[Tuple[multiprocessing.Queue, Dict[str, Any]]] = None
    if log_queue:
        pool_initializer_func = pool_worker_initializer_with_data
        initializer_args = (log_queue, init_args_for_worker)

    pool = None  # Initialize pool to None for finally block
    try:
        pool = Pool(
            processes=num_processes,
            initializer=pool_initializer_func,
            initargs=initializer_args,  # type: ignore[arg-type]
        )

        logger.info("Submitting tasks and processing results with imap_unordered...")
        imap_results = pool.imap_unordered(
            run_backtest_for_params, unique_params_for_backtest
        )

        processed_unique_count = 0
        successful_unique_count = 0
        total_results_to_process = unique_combinations_to_run_count

        # Define desired columns for CSV (can be moved to constants)
        DESIRED_RESULT_METRICS = [
            "cumulative_profit",
            "cumulative_profit_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "profit_factor",
            "max_drawdown_pct",
            "max_drawdown_abs",
            "longest_drawdown_duration",
            "max_consecutive_wins",
            "max_consecutive_losses",
            "total_trades",
            "winning_trades",
            "losing_trades",
            "win_rate",
            "average_trade_pnl",
            "average_winning_trade",
            "average_losing_trade",
        ]

        results_batch: List[Dict[str, Any]] = []
        BATCH_SIZE = 5000

        progress_iterator = tqdm(
            imap_results,
            total=total_results_to_process,
            desc=f"Optimizing {strategy_short_name}/{symbol}",
            unit=" combo",
            file=sys.stderr,
            disable=None,
            ncols=100,
        )

        for result_params, result_metrics in progress_iterator:
            processed_unique_count += 1

            if result_params is not None and result_metrics is not None:
                successful_unique_count += 1

                _ = result_metrics.pop("sharpe_ratio_ts", None)

                if save_details and details_filepath:
                    detail_entry = {f"param_{k}": v for k, v in result_params.items()}
                    detail_entry.update(
                        {f"result_{k}": v for k, v in result_metrics.items()}
                    )
                    results_batch.append(detail_entry)

                    if len(results_batch) >= BATCH_SIZE:
                        try:
                            details_df = pd.DataFrame(results_batch)
                            param_cols = [
                                c for c in details_df.columns if c.startswith("param_")
                            ]
                            result_cols = [
                                f"result_{m}" for m in DESIRED_RESULT_METRICS
                            ]
                            desired_cols = param_cols + result_cols
                            df_to_save = details_df.reindex(columns=desired_cols)
                            write_header = (
                                not details_filepath.exists()
                                or details_filepath.stat().st_size == 0
                            )
                            df_to_save.to_csv(
                                details_filepath,
                                mode="a",
                                header=write_header,
                                index=False,
                            )
                            results_batch.clear()
                            logger.debug(
                                f"Wrote batch of {len(details_df)} results to CSV: {details_filepath}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error writing batch to CSV file {details_filepath}: {e}"
                            )

                # --- Update Best Result *for this run* --- #
                current_metric_value = result_metrics.get(optimization_metric)
                if current_metric_value is not None:
                    is_initial_best = run_best_metric_value == float(
                        "inf"
                    ) or run_best_metric_value == -float("inf")
                    is_better = False
                    if pd.notna(current_metric_value):
                        if is_initial_best:
                            is_better = True
                        elif (
                            minimize_metric
                            and current_metric_value < run_best_metric_value
                        ):
                            is_better = True
                        elif (
                            not minimize_metric
                            and current_metric_value > run_best_metric_value
                        ):
                            is_better = True

                    if is_better:
                        run_best_metric_value = current_metric_value
                        run_best_params = result_params
                        run_best_metrics_dict = result_metrics
                        # Log new best for *this run* only if needed (can be verbose)
                        # logger.info(f"(Run) New best {optimization_metric}: {run_best_metric_value:.4f}")
            else:
                logger.warning("A backtest worker task returned None. Skipping result.")

        # Close pool and wait for workers
        pool.close()
        pool.join()
        logger.info("Worker pool closed and joined.")

        if total_results_to_process > 0:
            print(file=sys.stderr)  # Newline after tqdm

        # Write final batch if needed
        if save_details and results_batch and details_filepath:
            try:
                details_df = pd.DataFrame(results_batch)
                param_cols = [c for c in details_df.columns if c.startswith("param_")]
                result_cols = [f"result_{m}" for m in DESIRED_RESULT_METRICS]
                desired_cols = param_cols + result_cols
                df_to_save = details_df.reindex(columns=desired_cols)
                write_header = (
                    not details_filepath.exists()
                    or details_filepath.stat().st_size == 0
                )
                df_to_save.to_csv(
                    details_filepath, mode="a", header=write_header, index=False
                )
                logger.debug(
                    f"Wrote final batch of {len(details_df)} results to CSV: {details_filepath}"
                )
            except Exception as e:
                logger.error(f"Error writing final batch to CSV: {e}")

    except Exception as e:
        logger.exception(
            f"Error during parallel backtesting pool execution: {e}", exc_info=True
        )
        # Return None if the pool failed critically
        return None, None
    finally:
        # Ensure pool is terminated even if errors occurred before close/join
        if pool:
            pool.terminate()
            pool.join()

    logger.info(
        f"Parallel backtesting finished. Processed {processed_unique_count}/{total_results_to_process} results ({successful_unique_count} successful)."
    )

    # Return the best results found *during this specific run*
    return run_best_params, run_best_metrics_dict


# --- End Helper Function ---


# --- Helper Function: Find Overall Best from CSV --- << NEW >>
def _find_overall_best_from_csv(
    details_filepath: Path,
    optimization_metric: str,
    source_param_grid: Optional[Dict[str, List[Any]]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Reads the full optimization details CSV and finds the overall best parameters.

    Args:
        details_filepath: Path to the CSV file containing detailed results.
        optimization_metric: The metric to optimize for.
        source_param_grid: The original unprocessed parameter grid (used for type conversion).

    Returns:
        Tuple containing the overall best parameters and metrics dictionary found in the CSV.
        Returns (None, None) if the file is empty, the metric column is missing,
        or type conversion fails.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Analyzing full results file to determine overall best params: {details_filepath}"
    )

    # Determine if metric should be minimized
    minimize_metric = optimization_metric in ["max_drawdown"]

    try:
        if not details_filepath.is_file() or details_filepath.stat().st_size == 0:
            logger.warning(
                f"Details file {details_filepath} not found or is empty. Cannot determine overall best."
            )
            return None, None

        df_full_results = pd.read_csv(details_filepath, low_memory=False)

        if df_full_results.empty:
            logger.warning(
                f"Details file {details_filepath} is empty. Cannot determine overall best."
            )
            return None, None

        metric_col = f"result_{optimization_metric}"
        if metric_col not in df_full_results.columns:
            logger.warning(
                f"Metric column '{metric_col}' not found in {details_filepath}. Cannot determine overall best."
            )
            return None, None

        # Drop rows where the target metric is NaN
        df_valid_metric = df_full_results.dropna(subset=[metric_col])

        if df_valid_metric.empty:
            logger.warning(
                f"No valid (non-NaN) values found for metric '{metric_col}' in {details_filepath}. Cannot determine overall best."
            )
            return None, None

        if minimize_metric:
            best_idx = df_valid_metric[metric_col].idxmin()
        else:
            best_idx = df_valid_metric[metric_col].idxmax()

        best_row = df_valid_metric.loc[best_idx]

        # Extract param columns and values from the best row
        param_cols = [c for c in df_full_results.columns if c.startswith("param_")]
        csv_best_params_raw = {
            col.replace("param_", "", 1): best_row[col] for col in param_cols
        }

        # --- Convert CSV params back to correct types --- #
        type_map_from_source: Dict[str, Optional[type]] = {}
        if source_param_grid:  # Use the passed source grid
            for key, value_list in source_param_grid.items():
                if value_list:
                    # Simplified type inference (assumes simple types in grid)
                    type_map_from_source[key] = type(value_list[0])
                else:
                    type_map_from_source[key] = None  # No type info if list empty
        else:
            logger.warning(
                "Source parameter grid not provided for CSV analysis. Type conversion for overall best params might be inaccurate."
            )

        overall_best_params_typed: Dict[str, Any] = {}
        conversion_successful = True
        for k, v in csv_best_params_raw.items():
            if pd.isna(v):
                overall_best_params_typed[k] = None
                continue

            expected_type = type_map_from_source.get(k)
            csv_value_str = str(v).strip()

            if csv_value_str.lower() == "none":
                overall_best_params_typed[k] = None
                continue

            try:
                if expected_type == bool:
                    overall_best_params_typed[k] = csv_value_str.lower() in [
                        "true",
                        "1",
                        "t",
                        "y",
                        "yes",
                    ]
                elif expected_type == int:
                    overall_best_params_typed[k] = int(csv_value_str)
                elif expected_type == float:
                    overall_best_params_typed[k] = float(csv_value_str)
                elif (
                    expected_type is None or expected_type is object
                ):  # Fallback if type unknown or pandas object
                    # <<< SIMPLIFY FALLBACK INFERENCE >>>
                    # Basic inference - try int, float, bool, then default to string
                    try:
                        # Attempt int conversion (allow float strings like '1.0')
                        overall_best_params_typed[k] = int(float(v))
                    except (ValueError, TypeError):  # noqa: E722
                        try:
                            # Attempt float conversion
                            overall_best_params_typed[k] = float(v)
                        except (ValueError, TypeError):  # noqa: E722
                            # Attempt boolean conversion
                            if csv_value_str.lower() == "true":
                                overall_best_params_typed[k] = True
                            elif csv_value_str.lower() == "false":
                                overall_best_params_typed[k] = False
                            else:
                                # Fallback to the original string value if all else fails
                                overall_best_params_typed[k] = csv_value_str
                                logger.debug(
                                    f"Could not infer numeric/bool type for param '{k}'='{v}', keeping as string."
                                )
                else:
                    # Try direct conversion for other simple types if needed
                    try:
                        overall_best_params_typed[k] = expected_type(v)
                    except Exception as direct_conv_e:
                        logger.warning(
                            f"Direct conversion using expected type {expected_type} failed for '{k}': {direct_conv_e}. Falling back to string."
                        )
                        overall_best_params_typed[k] = (
                            csv_value_str  # Fallback to string
                        )

            except (ValueError, TypeError) as e:
                logger.error(
                    f"Failed to convert overall best param value '{v}' for key '{k}' to expected type {expected_type}: {e}. Skipping update."
                )
                conversion_successful = False
                break  # Exit inner loop on first conversion error for this row
        # --- End Type Conversion for one row --- #

        # Log the best parameters found
        logger.info(f"Overall best parameters found: {overall_best_params_typed}")

        # Return the best parameters and metrics
        return overall_best_params_typed, {
            optimization_metric: df_valid_metric[metric_col].iloc[best_idx]
        }

    except pd.errors.EmptyDataError:
        logger.warning(
            f"Details file {details_filepath} is empty during analysis. Cannot determine overall best."
        )
        return None, None
    except FileNotFoundError:
        logger.error(
            f"Details file {details_filepath} not found for final analysis. Cannot determine overall best."
        )
        return None, None
    except Exception as e:
        logger.error(
            f"Error reading or analyzing full results file {details_filepath}: {e}. Cannot determine overall best.",
            exc_info=True,
        )
        return None, None


# --- End Helper Function ---


def optimize_strategy(
    config_path: str,
    symbol: str,
    strategy_class_name: str,
    cmd_apply_atr_filter: bool,
    cmd_apply_seasonality_filter: bool,
    completed_param_reps: Optional[set] = None,
    save_details: bool = False,
    details_filepath: Optional[Path] = None,
    optimization_metric: str = "sharpe_ratio",
    num_processes: int = DEFAULT_OPT_PROCESSES,
    log_queue: Optional[multiprocessing.Queue] = None,
    shared_worker_args: Dict = {},
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    # Implementation of optimize_strategy function
    logger = logging.getLogger(__name__)
    # Ensure it's a set (should be passed correctly from main)
    completed_param_reps = completed_param_reps or set()

    # --- Ensure details_filepath exists if saving details ---
    if save_details:
        if details_filepath is None:
            logger.error("details_filepath must be provided when save_details is True.")
            return None, None
        try:
            details_filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Detailed results will be saved incrementally to: {details_filepath}"
            )
        except Exception as e:
            logger.error(f"Failed to create directory for {details_filepath}: {e}")
            return None, None
    # --- End Ensure Path --- #

    logger.info(
        f"Starting optimization for Strategy: {strategy_class_name}, Symbol: {symbol}"
    )
    strategy_class = STRATEGY_MAP.get(strategy_class_name)
    if not strategy_class:
        logger.error(
            f"Strategy name '{strategy_class_name}' not found in STRATEGY_MAP."
        )
        return None, None
    strategy_class_name = strategy_class.__name__

    # 1. Load and Prepare Grid
    try:
        config_file = Path(config_path)
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_file, "r") as f:
            full_config_data = yaml.safe_load(f)
            if not isinstance(full_config_data, dict):
                raise ValueError("Config file does not contain a dictionary.")

        param_grid, source_param_grid_for_types = _load_and_prepare_param_grid(
            full_config=full_config_data,
            symbol=symbol,
            strategy_class_name=strategy_class_name,
            cmd_apply_atr_filter=cmd_apply_atr_filter,
            cmd_apply_seasonality_filter=cmd_apply_seasonality_filter,
        )
        # Update source_param_grid needed for CSV analysis
        source_param_grid = source_param_grid_for_types

    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        logger.error(
            f"Failed to load or prepare param grid for {strategy_class_name} on {symbol}: {e}"
        )
        return None, None

    # 2. Generate Combinations (Handles Resume)
    unique_params_for_backtest = _generate_combinations(
        param_grid=param_grid,
        strategy_class_name=strategy_class_name,
        completed_param_reps=completed_param_reps,
    )
    unique_combinations_to_run_count = len(unique_params_for_backtest)

    # 3. Run Backtests (if needed)
    run_best_params: Optional[Dict] = None
    run_best_metrics_dict: Optional[Dict] = None
    if unique_combinations_to_run_count == 0:
        logger.warning(
            f"No parameter combinations left to run for {strategy_class_name} on {symbol} (either none generated or all were previously completed). Skipping backtesting."
        )
    else:
        # Determine number of processes (copied from before refactor)
        available_cpus = cpu_count()
        if num_processes > available_cpus:
            logger.warning(
                f"Requested processes ({num_processes}) exceed available CPUs ({available_cpus}). Using {available_cpus}."
            )
            num_processes = available_cpus
        elif num_processes <= 0:
            num_processes = available_cpus
            logger.info(f"Using default number of processes: {num_processes}")
        else:
            logger.info(f"Using specified number of processes: {num_processes}")

        run_best_params, run_best_metrics_dict = _run_parallel_backtests(
            unique_params_for_backtest=unique_params_for_backtest,
            num_processes=num_processes,
            log_queue=log_queue,
            shared_worker_args=shared_worker_args,
            save_details=save_details,
            details_filepath=details_filepath,
            optimization_metric=optimization_metric,
            strategy_short_name=strategy_class_name,
            symbol=symbol,
        )

    # 4. Determine Overall Best Params from Full History
    overall_best_params = run_best_params
    overall_best_metrics_dict = run_best_metrics_dict
    minimize_metric = optimization_metric in ["max_drawdown"]
    if run_best_params and run_best_metrics_dict:
        best_metric_value = run_best_metrics_dict.get(
            optimization_metric, float("inf") if minimize_metric else float("-inf")
        )
    else:
        best_metric_value = float("inf") if minimize_metric else float("-inf")

    if save_details and details_filepath:
        csv_best_params, csv_best_metrics = _find_overall_best_from_csv(
            details_filepath=details_filepath,
            optimization_metric=optimization_metric,
            source_param_grid=source_param_grid,
        )
        if csv_best_params is not None and csv_best_metrics is not None:
            csv_metric_value = csv_best_metrics.get(optimization_metric)
            is_csv_better = False
            if csv_metric_value is not None and pd.notna(csv_metric_value):
                if best_metric_value == float("inf") or best_metric_value == -float(
                    "inf"
                ):
                    is_csv_better = True
                elif minimize_metric and csv_metric_value < best_metric_value:
                    is_csv_better = True
                elif not minimize_metric and csv_metric_value > best_metric_value:
                    is_csv_better = True
            if is_csv_better:
                csv_log_val = (
                    f"{csv_metric_value:.4f}" if pd.notna(csv_metric_value) else "N/A"
                )
                run_log_val = (
                    f"{best_metric_value:.4f}"
                    if pd.notna(best_metric_value)
                    and best_metric_value not in [float("inf"), -float("inf")]
                    else "N/A"
                )
                logger.info(
                    f"Overall best from CSV ({optimization_metric}={csv_log_val}) is better than current run best ({optimization_metric}={run_log_val}). Using CSV result."
                )
                overall_best_params = csv_best_params
                overall_best_metrics_dict = csv_best_metrics
                best_metric_value = csv_metric_value
            else:
                logger.info(
                    "Best result from current run (if any) is better than or equal to best found in CSV. Keeping current run result."
                )
        else:
            logger.warning(
                "Could not determine overall best from CSV analysis. Using best from current run (if any)."
            )

    # 5. Final Logging & Return
    logger.info(f"Finished processing results for {strategy_class_name} on {symbol}.")
    if overall_best_params is None:
        logger.warning(
            f"No best parameters found for {strategy_class_name} on {symbol}. No successful backtests or specified metric '{optimization_metric}' not found?"
        )
    else:
        if (
            overall_best_metrics_dict
            and pd.notna(best_metric_value)
            and best_metric_value not in [float("inf"), -float("inf")]
        ):
            final_best_metric_value_str = f"{best_metric_value:.4f}"
        else:
            final_best_metric_value_str = "N/A"
        logger.info(
            f"Best parameters found for {strategy_class_name} on {symbol} (overall best from history if available): "
            # Need adjust_params_for_printing function defined below
            f"{adjust_params_for_printing(overall_best_params, strategy_class_name)} "
            f"with {optimization_metric} = {final_best_metric_value_str}"
        )

    return overall_best_params, overall_best_metrics_dict


# --- End optimize_strategy ---


# <<< ADD adjust_params_for_printing BACK >>>
def adjust_params_for_printing(
    params: Dict[str, Any], strategy_class_name: str
) -> Dict[str, Any]:
    """Converts specific strategy param representations for printing/saving.
    Currently handles LongShortStrategy tuple params.
    """
    printable_params = params.copy()
    if strategy_class_name == "LongShortStrategy":
        # Convert tuple back to low/high if they exist
        if (
            "return_thresh" in printable_params
            and isinstance(printable_params["return_thresh"], tuple)
            and len(printable_params["return_thresh"]) == 2
        ):
            rt = printable_params.pop("return_thresh")
            printable_params["return_thresh_low"] = rt[0]
            printable_params["return_thresh_high"] = rt[1]
        if (
            "volume_thresh" in printable_params
            and isinstance(printable_params["volume_thresh"], tuple)
            and len(printable_params["volume_thresh"]) == 2
        ):
            vt = printable_params.pop("volume_thresh")
            printable_params["volume_thresh_low"] = vt[0]
            printable_params["volume_thresh_high"] = vt[1]
    # Add adjustments for other strategies if needed
    return printable_params


# <<< END ADD FUNCTION >>>


def save_best_params(
    best_params: Dict[str, Any],
    strategy_class_name: str,
    details_filepath: Path,
    optimization_metric: str,
    best_metrics: Dict[str, Any],
):
    # Implementation of save_best_params function
    pass
