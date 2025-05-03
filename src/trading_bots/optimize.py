import pandas as pd
import numpy as np
import itertools
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Iterator, Tuple, Optional, cast, Set
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

    # Use a distinct logger name for worker init messages if needed
    worker_logger = logging.getLogger(
        f"WorkerInit.{multiprocessing.current_process().pid}"
    )  # Corrected

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
        worker_logger.error(
            "Worker did not receive data DataFrame during initialization."
        )  # Corrected
        # Attempting to log to queue if possible
        if worker_log_queue:
            try:
                err_record = worker_logger.makeRecord(  # Corrected
                    name=worker_logger.name,  # Corrected
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
        worker_logger.critical(f"Worker could not import strategies: {e}")  # Corrected
        if worker_log_queue:
            try:
                err_record = worker_logger.makeRecord(  # Corrected
                    name=worker_logger.name,  # Corrected
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
    logger_local = logging.getLogger(__name__)  # Corrected

    # Access globals
    global worker_data, worker_shared_args, WORKER_STRATEGY_MAP

    # +++ ADD DEBUG LOGGING FOR PARAMS ID +++
    logger_local.debug(  # Corrected
        f"Worker {multiprocessing.current_process().pid} ENTRY - Params object ID: {id(params)}"
    )
    # +++ END DEBUG LOGGING +++

    # Check if globals were initialized correctly
    if worker_data is None or not worker_shared_args or not WORKER_STRATEGY_MAP:
        logger_local.error(  # Corrected
            f"Worker {multiprocessing.current_process().pid} found uninitialized globals! Skipping task."
        )
        return None, None

    # --- Extract Base Config from Shared Args --- << MODIFIED >>
    backtest_config = None  # Initialize to handle potential errors below
    try:  # Added try block around config preparation
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
        strategy_class = WORKER_STRATEGY_MAP.get(strategy_short_name)
        if not strategy_class:
            logger_local.error(
                f"Strategy {strategy_short_name} not found in worker map."
            )  # Corrected
            return params, None
        sig = inspect.signature(strategy_class.__init__)  # type: ignore[misc]
        valid_init_params = {p for p in sig.parameters if p != "self"}
        run_config_data["strategy_params"] = {  # Indentation fixed
            k: v for k, v in params.items() if k in valid_init_params
        }  # Indentation fixed

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
            # Only override atr_filter_multiplier if grid value is not None
            if (
                "atr_filter_threshold" in params
                and params["atr_filter_threshold"] is not None
            ):
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
            logger_local.error(  # Corrected
                f"Failed to validate BacktestRunConfig for params {params}:\\n{e}"
            )
            return params, None

    except Exception as e:  # Added except block for the outer try
        logger_local.error(  # Corrected
            f"Error preparing BacktestRunConfig in worker for params {params}: {e}",
            exc_info=True,
        )
        return params, None

    # Ensure backtest_config was successfully created
    if backtest_config is None:
        logger_local.error(
            f"Worker {multiprocessing.current_process().pid}: backtest_config preparation failed. Cannot run backtest."
        )  # Corrected
        return params, None

    # --- Call Refactored run_backtest --- << MODIFIED >>
    try:  # Added try block around run_backtest call
        result_metrics = run_backtest(
            data=worker_data,  # Use global data
            config=backtest_config,  # Pass the config object
        )

        if not result_metrics:
            logger_local.warning(  # Corrected
                f"Worker {multiprocessing.current_process().pid}: run_backtest function returned None or empty metrics for params: {params}"
            )
            # Still return params, but empty results
            return params, None  # Return None for metrics

        # --- Remove detailed trade log BEFORE returning from worker ---
        if result_metrics and "result_performance_summary" in result_metrics:
            del result_metrics["result_performance_summary"]
            logger_local.debug(
                "Removed result_performance_summary from worker result."
            )  # Corrected
        # --- End removal ---

        # Return the original params (passed into function) and the results
        return params, result_metrics

    except Exception as e:  # Corrected: Added except block for backtest errors
        logger_local.error(  # Corrected
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
                else:  # Added else block
                    return (4, str(x))  # Fallback for other types - Fixed indentation

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

    return (
        param_grid,
        source_param_grid_for_types,
    )  # Ensure this is aligned with the 'def' line


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

            active_dependent_params_for_combo = {}
            fixed_dependent_params_for_combo = {}
            for flag_name, is_active in flag_combo_dict.items():
                deps = flags_and_deps.get(flag_name, [])
                for dep_name in deps:
                    if dep_name in dependent_params:
                        if is_active:
                            active_dependent_params_for_combo[dep_name] = (
                                dependent_params[dep_name]
                            )
                        else:
                            if dependent_params[dep_name]:  # Ensure list is not empty
                                fixed_dependent_params_for_combo[
                                    dep_name
                                ] = dependent_params[
                                    dep_name
                                ][  # Corrected indentation
                                    0
                                ]
                            else:  # Corrected indentation
                                fixed_dependent_params_for_combo[dep_name] = (
                                    None  # Corrected indentation
                                )

            active_dep_names = list(active_dependent_params_for_combo.keys())
            active_dep_values = list(active_dependent_params_for_combo.values())

            dependent_combinations = (
                itertools.product(*active_dep_values)
                if active_dependent_params_for_combo
                else [()]
            )

            for dep_combo_values in dependent_combinations:
                active_dep_combo_dict = dict(zip(active_dep_names, dep_combo_values))

                for indep_combo_values in independent_combinations:
                    indep_combo_dict = dict(zip(independent_names, indep_combo_values))
                    final_params = {
                        **flag_combo_dict,
                        **fixed_dependent_params_for_combo,
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

    # logger.info("Generating and deduplicating unique parameter combinations...") # Changed
    print(
        "INFO: Generating and deduplicating unique parameter combinations...",
        file=sys.stderr,
    )  # Changed
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
        log_interval = 10000  # Increased log interval
        if generated_count % log_interval == 0:
            logger.debug(f"Generated {generated_count} combinations so far...")

        if rep in unique_rep_to_params_map:
            continue
        unique_rep_to_params_map[rep] = params

    total_unique_combinations = len(unique_rep_to_params_map)
    # logger.info(f"Smart generator yielded {generated_count} combinations.") # Changed
    print(
        f"INFO: Smart generator yielded {generated_count} combinations.",
        file=sys.stderr,
    )  # Changed
    # logger.info( # Changed
    #     f"Deduplication complete: {total_unique_combinations} unique combinations identified initially."
    # )
    print(
        f"INFO: Deduplication complete: {total_unique_combinations} unique combinations identified initially.",
        file=sys.stderr,
    )  # Changed
    # <<< ADDED LOGGING >>>
    # logger.info(f"Total unique combinations calculated from CURRENT config: {total_unique_combinations}") # Changed
    print(
        f"INFO: Total unique combinations calculated from CURRENT config: {total_unique_combinations}",
        file=sys.stderr,
    )  # Changed
    # <<< END LOGGING >>>
    duplicate_count = generated_count - total_unique_combinations
    if duplicate_count > 0:
        # logger.info(f"  ({duplicate_count} duplicate combinations ignored)") # Changed
        print(
            f"INFO:   ({duplicate_count} duplicate combinations ignored)",
            file=sys.stderr,
        )  # Changed

    # --- Filter based on completed_param_reps (Resume) --- #
    unique_params_list = list(unique_rep_to_params_map.values())
    if not completed_param_reps:
        # logger.info(f"Running all {total_unique_combinations} unique combinations.") # Changed
        print(
            f"INFO: Running all {total_unique_combinations} unique combinations.",
            file=sys.stderr,
        )  # Changed
        # DEBUG: Print first few generated reps when not resuming
        for i, p in enumerate(unique_params_list[:5]):
            print(
                f"DEBUG GENERATE[{i}]: Generated rep = {repr(params_to_tuple_rep(p))}",
                file=sys.stderr,
            )
        # END DEBUG
        return unique_params_list
    else:
        params_to_run = []
        skipped_count = 0
        logger.debug(
            f"--- Resuming: Comparing {total_unique_combinations} generated params against {len(completed_param_reps)} loaded reps ---"
        )
        for i, params in enumerate(unique_params_list):  # Added index i for debug
            rep = params_to_tuple_rep(params)
            # DEBUG: Print first few generated reps when resuming for comparison
            if i < 5:
                print(
                    f"DEBUG GENERATE[{i}]: Generated rep = {repr(rep)}", file=sys.stderr
                )
            # END DEBUG
            if rep in completed_param_reps:
                skipped_count += 1
            else:
                params_to_run.append(params)
        logger.debug(f"--- Resume Check Finished: Total Skipped={skipped_count} ---")

        remaining_count = len(params_to_run)
        # logger.info(f"Resume: Skipped {skipped_count} already completed parameters.") # Changed
        print(
            f"INFO: Resume: Skipped {skipped_count} already completed parameters.",
            file=sys.stderr,
        )  # Changed
        # logger.info(f"Resume: {remaining_count} parameters remaining to run.") # Changed
        print(
            f"INFO: Resume: {remaining_count} parameters remaining to run.",
            file=sys.stderr,
        )  # Changed
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
    strategy_short_name: str,  # <-- Already accepts short name
    symbol: str,
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
    # Ensure the correct strategy name (short name) is passed for worker lookup
    init_args_for_worker["strategy_short_name"] = (
        strategy_short_name  # <-- Use the passed short name
    )
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

        # <<< ADDED LOGGING >>>
        logger.info(
            f"Initializing tqdm progress bar with total={total_results_to_process}"
        )
        # <<< END LOGGING >>>
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
                    # Create dictionary with individual param_ columns
                    detail_entry = {f"param_{k}": v for k, v in result_params.items()}
                    # Add the JSON string representation to the 'parameters' column
                    try:
                        detail_entry["parameters"] = json.dumps(result_params)
                    except TypeError as e:
                        logger.warning(
                            f"Could not serialize parameters to JSON: {result_params} - Error: {e}"
                        )
                        detail_entry["parameters"] = None  # Or handle error differently

                    # Add result_ columns
                    detail_entry.update(
                        {f"result_{k}": v for k, v in result_metrics.items()}
                    )
                    results_batch.append(detail_entry)

                    if len(results_batch) >= BATCH_SIZE:
                        try:
                            details_df = pd.DataFrame(results_batch)
                            # Define desired columns dynamically + add 'parameters'
                            param_cols = sorted(
                                [
                                    c
                                    for c in details_df.columns
                                    if c.startswith("param_")
                                ]
                            )
                            result_cols = sorted(
                                [
                                    f"result_{m}"
                                    for m in DESIRED_RESULT_METRICS
                                    if f"result_{m}" in details_df.columns
                                ]
                            )
                            # Ensure 'parameters' column is included and define the order
                            desired_cols = param_cols + ["parameters"] + result_cols

                            # Reindex and handle potential missing columns gracefully
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
                # This block should be OUTSIDE the `if save_details:` block but INSIDE the `if result_params is not None:` block
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
            else:  # This else corresponds to `if result_params is not None and result_metrics is not None:`
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
                # Define desired columns dynamically + add 'parameters' (same logic as above)
                param_cols = sorted(
                    [c for c in details_df.columns if c.startswith("param_")]
                )
                result_cols = sorted(
                    [
                        f"result_{m}"
                        for m in DESIRED_RESULT_METRICS
                        if f"result_{m}" in details_df.columns
                    ]
                )
                # Ensure 'parameters' column is included and define the order
                desired_cols = param_cols + ["parameters"] + result_cols

                # Reindex and handle potential missing columns gracefully
                df_to_save = details_df.reindex(columns=desired_cols)

                write_header = (
                    not details_filepath.exists()
                    or details_filepath.stat().st_size == 0
                )
                df_to_save.to_csv(
                    details_filepath, mode="a", header=write_header, index=False
                )
                logger.debug(  # Indentation fixed
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
        if "pool" in locals() and pool:  # Check if pool exists before terminating
            pool.terminate()
            pool.join()

        logger.info(  # Indentation fixed
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
            logger.warning(  # Indentation fixed
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
            return None, None  # Indentation fixed

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
                            else:  # Indentation fixed
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
            optimization_metric: best_row[metric_col]  # Use best_row directly
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
    except Exception as e:  # Indentation fixed
        logger.error(
            f"Error reading or analyzing full results file {details_filepath}: {e}. Cannot determine overall best.",
            exc_info=True,
        )
        return None, None


# --- End Helper Function ---


def optimize_strategy(
    config_path: str,
    symbol: str,
    strategy_class_name: str,  # Use full class name internally
    strategy_short_name: str,  # Add short name argument
    cmd_apply_atr_filter: bool,
    cmd_apply_seasonality_filter: bool,
    completed_param_reps: Optional[Set[tuple]] = None,  # More specific set type
    save_details: bool = False,
    details_filepath: Optional[Path] = None,
    optimization_metric: str = "sharpe_ratio",
    num_processes: int = DEFAULT_OPT_PROCESSES,
    log_queue: Optional[multiprocessing.Queue] = None,
    shared_worker_args: Dict[str, Any] = {},  # Specific Dict type
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
            # Moved logger info inside the try after successful directory creation/check
            logger.info(
                f"Detailed results will be saved incrementally to: {details_filepath}"
            )
        except Exception as e:  # Added missing except block
            logger.error(f"Failed to create directory for {details_filepath}: {e}")
            return None, None
    # --- End Ensure Path --- #

    logger.info(
        f"Starting optimization for Strategy: {strategy_class_name}, Symbol: {symbol}"
    )
    # No need to lookup class, strategy_class_name is already the full name

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
            strategy_class_name=strategy_class_name,  # Use full name for loading from config
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
    logger.info("Generating parameter combinations...")  # Added log
    unique_params_for_backtest = _generate_combinations(
        param_grid=param_grid,
        strategy_class_name=strategy_class_name,  # Use full name for specific handling if needed
        completed_param_reps=completed_param_reps,
    )
    unique_combinations_to_run_count = len(unique_params_for_backtest)
    logger.info(
        f"Generated {unique_combinations_to_run_count} unique combinations to run."
    )  # Added log

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

        logger.info(
            f"Starting parallel backtest run for {unique_combinations_to_run_count} combinations..."
        )  # Added log
        run_best_params, run_best_metrics_dict = _run_parallel_backtests(
            unique_params_for_backtest=unique_params_for_backtest,
            num_processes=num_processes,
            log_queue=log_queue,
            shared_worker_args=shared_worker_args,
            save_details=save_details,
            details_filepath=details_filepath,
            optimization_metric=optimization_metric,
            strategy_short_name=strategy_short_name,  # Pass the short name here
            symbol=symbol,
        )
        logger.info("Parallel backtest run finished.")  # Added log

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
        logger.info(
            f"Analyzing details file ({details_filepath}) for overall best parameters..."
        )  # Added log
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
                # Indentation fixed for this logger call
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
    else:  # Fixed indentation for the corresponding else
        logger.info(
            "Skipping analysis of details file (save_details=False or details_filepath is None)."
        )  # Corrected log message

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
    details_filepath: Optional[Path],  # Allow None
    optimization_metric: str,
    best_metrics: Dict[str, Any],
    output_filepath: Path,  # Add the missing parameter
):
    # Implementation of save_best_params function
    pass  # Placeholder - actual implementation needed


# <<< START ADDED HELPER FUNCTION >>>
def get_details_filepath(
    strategy_short_name: str, symbol: str, start_date_str: str, end_date_str: str
) -> Path:
    """Generates a default filepath for the optimization details CSV."""
    # Sanitize components to be safe for filenames
    safe_strategy = sanitize_filename(strategy_short_name)
    safe_symbol = sanitize_filename(symbol)
    safe_start = sanitize_filename(start_date_str)
    safe_end = sanitize_filename(end_date_str)

    # Define the directory structure
    details_dir = Path("results") / "optimize" / "details"
    # Create the directory if it doesn't exist (handled in main, but good practice)
    # details_dir.mkdir(parents=True, exist_ok=True) # Optionally ensure here too

    # Construct the filename
    filename = f"{safe_strategy}_{safe_symbol}_{safe_start}_{safe_end}_details.csv"

    return details_dir / filename


# <<< END ADDED HELPER FUNCTION >>>


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trading strategy optimization.")
    # --- Add Arguments --- < RESTORED >
    parser.add_argument(
        "--strategy",
        required=True,
        help="Short name of the strategy class to optimize (e.g., MACross).",
        choices=STRATEGY_MAP.keys(),
    )
    parser.add_argument(
        "--symbol", required=True, help="Trading symbol (e.g., BTCUSDT)."
    )
    parser.add_argument(
        "--file", required=True, help="Path to the historical data CSV file."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the optimization parameters YAML config file.",
    )
    parser.add_argument(
        "--output-config", help="Path to save the best parameters YAML file."
    )
    parser.add_argument(
        "--opt-start",
        required=True,
        help="Start date for optimization period (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--opt-end",
        required=True,
        help="End date for optimization period (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--balance", type=float, required=True, help="Initial backtest balance."
    )
    parser.add_argument(
        "--commission",
        type=float,
        required=True,
        help="Commission per trade in basis points (e.g., 7.5 for 0.075%).",
    )
    parser.add_argument(
        "--metric",
        default="sharpe_ratio",
        help="Metric to optimize for.",
        choices=[
            "cumulative_profit",
            "sharpe_ratio",
            "profit_factor",
            "max_drawdown",
            "win_rate",
            "sortino_ratio",
            "longest_drawdown_duration",
        ],
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save detailed results of each combination to a CSV file.",
    )
    parser.add_argument(
        "--details-file",
        help="Optional specific path for the detailed results CSV file. Defaults to auto-generated name.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=DEFAULT_OPT_PROCESSES,
        help="Number of processes for parallel backtesting.",
    )
    # Filter Arguments
    parser.add_argument(
        "--apply-atr-filter",
        action="store_true",
        help="Enable ATR volatility filter globally during optimization.",
    )
    parser.add_argument(
        "--atr-filter-period",
        type=int,
        default=14,
        help="ATR period for the filter.",
    )
    parser.add_argument(
        "--atr-filter-multiplier",
        type=float,
        default=1.5,
        help="ATR multiplier for the filter threshold.",
    )
    parser.add_argument(
        "--atr-filter-sma-period",
        type=int,
        default=100,
        help="SMA period for ATR filter baseline (0 to disable SMA).",
    )
    parser.add_argument(
        "--apply-seasonality-filter",
        action="store_true",
        help="Enable seasonality filter globally during optimization.",
    )
    parser.add_argument(
        "--allowed-trading-hours-utc",
        type=str,
        default="",
        help="Allowed trading hours in UTC (e.g., '5-17'). Empty means all hours.",
    )
    parser.add_argument(
        "--apply-seasonality-to-symbols",
        type=str,
        default="",
        help="Comma-separated symbols to apply seasonality filter to. Empty applies to main symbol.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimization by loading completed parameters from details file.",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    # --- End Add Arguments --- </RESTORED >

    args = parser.parse_args()  # <-- CRITICAL LINE RESTORED

    print(f"DEBUG: Parsed args: {args}", file=sys.stderr)  # Added DEBUG

    # --- Logging Setup ---
    log_queue: multiprocessing.Queue = multiprocessing.Queue(
        -1
    )  # Central queue for logs, Added type hint
    log_level = getattr(logging, args.log.upper(), logging.INFO)

    # Initialize listener variable before try blocks
    log_listener: Optional[QueueListener] = None

    # Basic formatter for console output (main process only)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # More detailed formatter for the listener (includes process info)
    listener_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s [%(processName)s] - %(message)s"
    )

    # Setup root logger handler ONLY for the main process console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    # Filter so console handler only shows logs from the main process if needed,
    # or adjust levels as necessary. For now, let the root logger level control it.

    # Configure the root logger IN THE MAIN PROCESS
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Explicitly set the logging level for the 'src.trading_bots.optimize' logger in the main function.
    logging.getLogger("src.trading_bots.optimize").setLevel(log_level)

    # Remove existing handlers to avoid duplication if re-running in same session/notebook
    # Check before clearing: only clear if handlers already exist
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add ONLY the console handler to the root logger in the main process
    # Logs from worker processes will be handled by the listener separately
    root_logger.addHandler(console_handler)

    # --- Logging Listener Setup ---
    # Define listener handler (can be stdout or a file handler)
    # Here we use stdout, formatted with process info
    listener_console_handler = logging.StreamHandler(sys.stdout)
    listener_console_handler.setFormatter(listener_formatter)

    # Create the listener which runs in a separate thread
    log_listener = QueueListener(
        log_queue, listener_console_handler, respect_handler_level=True
    )
    log_listener.start()

    # Worker processes will use QueueHandler to send logs to the listener
    # The setup for worker logging (QueueHandler) happens within setup_worker_logging

    logger.info("Logging configured. Log level: %s", args.log.upper())
    # print(f"DEBUG: Logging configured. Log level: {args.log.upper()}", file=sys.stderr) # DEBUG print removed

    # --- Load Data ---
    print(
        f"DEBUG: Attempting to load data file: {args.file}", file=sys.stderr
    )  # Added DEBUG
    try:
        data = load_csv_data(str(Path(args.file)))  # Convert Path to str
        if data is None or data.empty:
            logger.error(f"Failed to load data or data is empty: {args.file}")
            if log_listener:
                log_listener.stop()  # Check if listener exists before stopping
            sys.exit(1)
        print(
            f"DEBUG: Successfully loaded data. Shape: {data.shape}", file=sys.stderr
        )  # Added DEBUG
    except Exception as e:  # Added missing except block and fixed indentation
        logger.error(
            f"CRITICAL ERROR loading data file {args.file}: {e}", exc_info=True
        )
        print(
            f"DEBUG: Exiting due to data loading error.", file=sys.stderr
        )  # Added DEBUG
        if log_listener:
            log_listener.stop()  # Check if listener exists before stopping
        sys.exit(1)
    # --- End Load Data ---

    # --- Strategy Check ---
    strategy_class = STRATEGY_MAP.get(args.strategy)
    if not strategy_class:
        logger.error(
            f"Strategy name '{args.strategy}' not found in STRATEGY_MAP. Available: {list(STRATEGY_MAP.keys())}"
        )
        if log_listener:
            log_listener.stop()  # Check if listener exists before stopping
        sys.exit(1)  # Fixed indentation
    strategy_class_name = strategy_class.__name__  # Get the actual class name
    print(
        f"DEBUG: Strategy '{args.strategy}' mapped to class '{strategy_class_name}'.",
        file=sys.stderr,
    )  # Added DEBUG & use mapped name
    # --- End Strategy Check ---

    # --- Load and Validate Config ---
    print(
        f"DEBUG: Attempting to load config file: {args.config}", file=sys.stderr
    )  # Added DEBUG
    try:
        config_file = Path(args.config)
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(config_file, "r") as f:
            raw_config_data = yaml.safe_load(f)
            if not isinstance(raw_config_data, dict):
                raise ValueError("Config file is not a dictionary.")
        # Validate using Pydantic
        config = OptimizeParamsConfig.model_validate(raw_config_data)
        print(
            f"DEBUG: Successfully loaded and validated config.", file=sys.stderr
        )  # Added DEBUG

    except (FileNotFoundError, yaml.YAMLError, ValidationError, ValueError) as e:
        logger.error(
            f"CRITICAL ERROR loading/validating config {args.config}: {e}",
            exc_info=True,
        )
        print(
            f"DEBUG: Exiting due to config loading/validation error.", file=sys.stderr
        )  # Added DEBUG
        if log_listener:
            log_listener.stop()  # Check if listener exists before stopping
        sys.exit(1)
    # --- End Load and Validate Config ---

    # --- Determine Details Filepath ---
    final_details_filepath: Optional[Path] = None
    if args.save_details:
        if args.details_file:
            final_details_filepath = Path(args.details_file)
        else:  # Fixed indentation
            # Use helper to generate default path if none provided
            try:
                start_dt = pd.to_datetime(args.opt_start).strftime("%Y%m%d")
                end_dt = pd.to_datetime(args.opt_end).strftime("%Y%m%d")
                final_details_filepath = get_details_filepath(
                    args.strategy, args.symbol, start_dt, end_dt
                )
            except Exception as e:
                logger.error(
                    f"Failed to generate default details filepath: {e}", exc_info=True
                )
                if log_listener:
                    log_listener.stop()  # Check if listener exists before stopping
                sys.exit(1)
        # Ensure directory exists if saving details
        if final_details_filepath:
            try:
                final_details_filepath.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(
                    f"Failed to create directory for details file {final_details_filepath}: {e}",
                    exc_info=True,
                )
                if log_listener:
                    log_listener.stop()  # Check if listener exists before stopping
                sys.exit(1)

    # --- Resume Logic ---
    completed_param_reps: set = set()
    if (
        args.resume
        and args.save_details
        and final_details_filepath
        and final_details_filepath.exists()
    ):
        logger.info(
            f"Resume flag set. Attempting to load completed params from {final_details_filepath}"
        )
        try:
            # --- Load current param grid keys for comparison --- <<< NEW >>>
            current_param_grid_keys = set()
            try:
                # Need the validated config_data from earlier
                config_file = Path(args.config)
                with open(config_file, "r") as f:
                    full_config_data = yaml.safe_load(
                        f
                    )  # Assuming raw_config_data holds the validated dict

                # Call helper to get the processed grid *for this run*
                param_grid_for_keys, _ = _load_and_prepare_param_grid(
                    full_config=full_config_data,  # Use the validated config
                    symbol=args.symbol,
                    strategy_class_name=strategy_class_name,  # Use mapped name
                    cmd_apply_atr_filter=args.apply_atr_filter,  # Fixed indentation
                    cmd_apply_seasonality_filter=args.apply_seasonality_filter,
                )
                current_param_grid_keys = set(
                    param_grid_for_keys.keys()
                )  # Fixed indentation
                logger.debug(
                    f"Resume Check: Current expected param keys = {current_param_grid_keys}"
                )
            except Exception as grid_load_err:  # Added except for inner try
                logger.error(
                    f"Resume ERROR: Could not load current param grid to determine keys: {grid_load_err}. Cannot reliably resume."
                )  # Fixed indentation
                # Clear completed reps to force full run if keys cannot be determined
                completed_param_reps.clear()  # Fixed indentation
                # Skip reading the CSV file
                raise grid_load_err  # Re-raise to trigger the outer exception handling # Fixed indentation
            # --- End loading current keys --- <<< END NEW >>>

            # Filter DtypeWarning specifically for this read_csv operation
            import warnings
            from pandas.errors import DtypeWarning

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DtypeWarning)
                existing_results_df = pd.read_csv(
                    final_details_filepath, low_memory=False
                )

            if (
                "parameters" in existing_results_df.columns and current_param_grid_keys
            ):  # Check if keys were loaded
                loaded_count = 0
                skipped_decode_errors = 0
                skipped_missing_key_errors = 0  # Count errors during key filling
                print(
                    f"DEBUG RESUME: Processing {len(existing_results_df['parameters'].dropna())} rows from CSV parameters column.",
                    file=sys.stderr,
                )  # Added DEBUG
                for idx, params_str in (
                    existing_results_df["parameters"].dropna().items()
                ):  # Use items() to get index
                    # print(f"DEBUG RESUME[{idx}]: Raw CSV params_str = {params_str[:200]}...", file=sys.stderr) # Commented out
                    try:
                        params_dict = json.loads(params_str)
                        # print(f"DEBUG RESUME[{idx}]: Loaded params_dict = {repr(params_dict)}", file=sys.stderr) # Commented out

                        # --- Check for missing keys --- <<< MODIFIED >>>
                        loaded_keys = set(params_dict.keys())
                        missing_keys = current_param_grid_keys - loaded_keys
                        if missing_keys:
                            logger.error(
                                f"Resume ERROR on row {idx}: Loaded parameters {repr(params_dict)} are missing expected keys from current config: {missing_keys}. Halting execution."
                            )
                            # Raise error instead of filling
                            raise ValueError(
                                f"Inconsistent parameters found in resume file row {idx}. Missing keys: {missing_keys}"
                            )

                        # Check for unexpected extra keys (optional, but good practice)
                        extra_keys = loaded_keys - current_param_grid_keys
                        if extra_keys:
                            logger.warning(
                                f"Resume WARNING on row {idx}: Loaded parameters {repr(params_dict)} have extra keys not in current config: {extra_keys}. Proceeding, but this might indicate inconsistency."
                            )
                        # --- End key check --- <<< END MODIFIED >>>

                        # Generate rep using the validated dictionary (which must have all keys now)
                        rep = params_to_tuple_rep(params_dict)
                        # print(f"DEBUG RESUME[{idx}]: Generated rep = {repr(rep)}", file=sys.stderr) # Commented out
                        completed_param_reps.add(rep)  # Fixed indentation
                        loaded_count += 1
                    except json.JSONDecodeError:
                        skipped_decode_errors += 1
                        logger.error(
                            f"JSONDecodeError on row {idx}: {params_str[:100]}..."
                        )  # Log errors with index
                    # Catch potential errors during key filling (less likely) or tuple conversion
                    except Exception as load_err:
                        skipped_missing_key_errors += 1
                        logger.error(
                            f"Error processing parameter tuple (or filling keys) from CSV row {idx} '{params_str[:100]}...': {load_err}"
                        )

                logger.info(
                    f"Successfully loaded {loaded_count} completed parameter representations."
                )
                if skipped_decode_errors > 0:
                    logger.warning(
                        f"Skipped {skipped_decode_errors} rows due to invalid JSON in 'parameters' column."
                    )
                if skipped_missing_key_errors > 0:
                    logger.warning(
                        f"Skipped {skipped_missing_key_errors} rows due to errors during parameter processing/key filling."
                    )
            elif not current_param_grid_keys:
                logger.warning(
                    "Could not determine current parameter keys. Skipping resume based on parameters column."
                )
            else:  # 'parameters' column missing
                logger.warning(
                    f"'parameters' column not found in {final_details_filepath}. Cannot resume based on parameters."
                )
        except FileNotFoundError:  # Added except for outer try
            logger.warning(
                f"Resume file {final_details_filepath} not found, running all combinations."
            )
        except Exception as e:  # Added except for outer try
            logger.error(
                f"Error reading resume file or loading grid keys {final_details_filepath}: {e}",
                exc_info=True,
            )
            logger.warning("Proceeding without resuming due to error.")
            completed_param_reps.clear()  # Ensure it's empty if error occurred
    print(
        f"DEBUG: Resume active: {args.resume}. Completed reps loaded: {len(completed_param_reps)}",
        file=sys.stderr,
    )  # Added DEBUG
    # --- End Resume Logic ---

    # --- Prepare Shared Args for Workers ---
    print(f"DEBUG: Preparing shared worker args.", file=sys.stderr)  # Added DEBUG
    shared_worker_args = {  # Fixed indentation
        "data": data,  # Pass the loaded DataFrame
        "symbol": args.symbol,
        "initial_balance": args.balance,
        "commission_bps": args.commission,
        # Use getattr for optional args with defaults if necessary
        "units": getattr(
            args, "units", 1.0
        ),  # Example: assuming 'units' might be optional
        # --- Filters ---
        "apply_atr_filter": args.apply_atr_filter,
        "atr_filter_period": args.atr_filter_period,
        "atr_filter_multiplier": args.atr_filter_multiplier,
        "atr_filter_sma_period": args.atr_filter_sma_period,
        "apply_seasonality_filter": args.apply_seasonality_filter,
        "allowed_trading_hours_utc": args.allowed_trading_hours_utc,
        "apply_seasonality_to_symbols": args.apply_seasonality_to_symbols,
    }
    # --- End Prepare Shared Args ---

    # --- Run Optimization ---
    strategy_short_name = args.strategy  # Get the original short name
    print(
        f"DEBUG: Calling optimize_strategy for {strategy_class_name} on {args.symbol}",
        file=sys.stderr,
    )  # Added DEBUG
    try:
        best_params, best_metrics_dict = optimize_strategy(
            config_path=args.config,
            symbol=args.symbol,
            strategy_class_name=strategy_class_name,  # Use mapped class name
            strategy_short_name=strategy_short_name,  # Pass short name
            cmd_apply_atr_filter=args.apply_atr_filter,
            cmd_apply_seasonality_filter=args.apply_seasonality_filter,
            completed_param_reps=completed_param_reps,
            save_details=args.save_details,
            details_filepath=final_details_filepath,
            optimization_metric=args.metric,
            num_processes=args.processes,
            log_queue=log_queue,
            shared_worker_args=shared_worker_args,
        )
    except Exception as opt_exc:
        logger.critical(
            f"Unhandled exception during optimize_strategy call: {opt_exc}",
            exc_info=True,
        )
        best_params = None
        best_metrics_dict = None
        print(
            f"DEBUG: Exiting due to unhandled exception in optimize_strategy.",
            file=sys.stderr,
        )  # Added DEBUG
        # Optionally re-raise or handle differently
    finally:
        # Ensure listener is stopped even if optimize_strategy fails
        pass  # Listener stopped at the very end

    print(f"DEBUG: optimize_strategy returned.", file=sys.stderr)  # Added DEBUG
    # --- End Run Optimization ---

    # --- Save Best Params ---
    if args.output_config and best_params and best_metrics_dict:
        print(
            f"DEBUG: Saving best parameters to {args.output_config}", file=sys.stderr
        )  # Added DEBUG
        try:
            output_path = Path(args.output_config)
            # Use the helper function to save
            save_best_params(
                best_params=best_params,
                strategy_class_name=strategy_class_name,
                details_filepath=final_details_filepath,  # Pass directly (now Optional in def)
                optimization_metric=args.metric,
                best_metrics=best_metrics_dict,
                output_filepath=output_path,  # Pass target output path
            )
            logger.info(
                f"Successfully saved best parameters to {output_path}"
            )  # Added success log
        except Exception as e:  # Added except block
            logger.error(
                f"Failed to save best parameters to {args.output_config}: {e}",
                exc_info=True,
            )  # Fixed indentation
            print(
                f"DEBUG: Error saving best parameters.", file=sys.stderr
            )  # Added DEBUG
    elif (
        args.output_config
    ):  # Fixed indentation and structure (elif instead of separate if)
        logger.warning(
            f"No best parameters found or an error occurred. Cannot save to {args.output_config}"
        )  # Fixed indentation
        print(f"DEBUG: No best parameters to save.", file=sys.stderr)  # Added DEBUG
    # --- End Save Best Params ---

    # --- Final Cleanup ---
    print(f"DEBUG: Stopping log listener.", file=sys.stderr)  # Added DEBUG
    if log_listener:  # Check if listener exists before stopping
        log_listener.stop()
    print(f"DEBUG: Script finished.", file=sys.stderr)  # Added DEBUG


if __name__ == "__main__":
    main()
