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
from functools import partial # Added for partial function application
import json # Added for stable serialization of params
from logging.handlers import QueueHandler, QueueListener # Added for multiprocessing logging
import multiprocessing # Added for multiprocessing
import logging.handlers
import time
import traceback # For listener error reporting
import re # Import re for sanitize_filename if not already imported

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
WORKER_STRATEGY_MAP: Dict[str, type] = {} # Map populated in initializer
# <<< End Worker Globals >>>

def worker_log_configurer(log_queue: multiprocessing.Queue):
    """Configures logging for a worker process to send ONLY to the queue."""
    root = logging.getLogger() 
    # Ensure root logger exists
    if not root:
        print(f"Worker {multiprocessing.current_process().pid}: Root logger not found!", file=sys.stderr)
        return

    # *** IMPORTANT: Remove existing handlers from worker's root logger ***
    if root.hasHandlers():
        print(f"Worker {multiprocessing.current_process().pid}: Removing {len(root.handlers)} existing handler(s) from root logger.", file=sys.stderr)
        root.handlers.clear()

    # Add ONLY the QueueHandler
    queue_handler = QueueHandler(log_queue)
    root.addHandler(queue_handler)
    # Set the level for this specific handler - messages BELOW this level won't be put on the queue
    # The overall root level will be set by the main process setup.
    root.setLevel(logging.DEBUG) # Send DEBUG and higher from workers to the queue

def pool_worker_initializer_with_data(log_q: multiprocessing.Queue, 
                                      shared_worker_args_for_init: Dict):
    """Initializer for worker processes.
       Sets up logging and stores shared data in global variables.
    """
    global worker_log_queue, worker_data, worker_shared_args, WORKER_STRATEGY_MAP
    
    print(f"Initializing worker {multiprocessing.current_process().pid} with queue {id(log_q)}...", file=sys.stderr) # Basic print for debug
    
    # 1. Configure Logging
    worker_log_queue = log_q
    worker_log_configurer(worker_log_queue)
    
    # 2. Store Shared Data/Args
    # Make a copy to ensure each worker has its own reference (though DataFrame might still share memory)
    worker_shared_args = shared_worker_args_for_init.copy()
    worker_data = worker_shared_args.pop('data', None) # Extract data to separate global
    if not isinstance(worker_data, pd.DataFrame) or worker_data.empty:
         print(f"CRITICAL ERROR in worker {multiprocessing.current_process().pid}: Did not receive valid data in initializer! Type: {type(worker_data)}", file=sys.stderr)
         # Maybe raise an error or log to stderr? Pool might handle worker exit.
         logger = logging.getLogger("WorkerInitError")
         logger.error("Worker did not receive data DataFrame during initialization.")
         # Attempting to log to queue if possible
         if worker_log_queue:
             try:
                 err_record = logger.makeRecord(name="WorkerInitError", level=logging.CRITICAL, 
                                               fn="", lno=0, msg="Worker data is None", args=[], 
                                               exc_info=None, func="pool_worker_initializer_with_data")
                 worker_log_queue.put(err_record)
             except Exception as e:
                 print(f"WorkerInitError: Failed to put log record in queue: {e}", file=sys.stderr)
         # Exiting worker might be necessary if data is crucial
         # sys.exit(1) # Or let the pool handle it

    # 3. Populate Worker Strategy Map (avoids repeated imports in worker function)
    try:
        from .strategies import (
            LongShortStrategy, MovingAverageCrossoverStrategy, 
            RsiMeanReversionStrategy, BollingerBandReversionStrategy
        )
        WORKER_STRATEGY_MAP = {
            "LongShort": LongShortStrategy,
            "MACross": MovingAverageCrossoverStrategy,
            "RSIReversion": RsiMeanReversionStrategy,
            "BBReversion": BollingerBandReversionStrategy,
        }
    except ImportError as e:
        print(f"CRITICAL ERROR in worker {multiprocessing.current_process().pid}: Could not import strategies: {e}")
        # Log/Handle error similar to data error
        logger = logging.getLogger("WorkerInitError")
        logger.critical(f"Worker could not import strategies: {e}")
        if worker_log_queue:
            try:
                 err_record = logger.makeRecord(name="WorkerInitError", level=logging.CRITICAL, 
                                               fn="", lno=0, msg=f"Strategy import failed: {e}", args=[], 
                                               exc_info=None, func="pool_worker_initializer_with_data")
                 worker_log_queue.put(err_record)
            except Exception as log_e:
                print(f"WorkerInitError: Failed to put log record in queue: {log_e}", file=sys.stderr)
        # sys.exit(1) 

    print(f"Worker {multiprocessing.current_process().pid} initialized successfully. Data shape: {worker_data.shape if worker_data is not None else 'None'}", file=sys.stderr)


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
    items = []
    logger = logging.getLogger(__name__) # Use logger defined in the module
    for k, v in sorted(params.items()):
        # Handle floats specifically for rounding
        if isinstance(v, float):
            try:
                rounded_v = round(v, precision)
                if rounded_v == -0.0:
                    rounded_v = 0.0
                items.append((k, rounded_v))
            except (ValueError, TypeError) as e:
                 logger.warning(f"Could not round float parameter '{k}' value {repr(v)}: {e}. Using original value.")
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
                        rounded_tuple_elements.append(elem) # Keep non-floats as is
                items.append((k, tuple(rounded_tuple_elements)))
            except (ValueError, TypeError) as e:
                 logger.warning(f"Could not round float elements in tuple parameter '{k}' value {repr(v)}: {e}. Using original value.")
                 items.append((k, v)) # Use original tuple on error
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
                 items.append((k, tuple(sorted(rounded_list_elements, key=lambda x: (isinstance(x, type(None)), x) if isinstance(x, (int, float, str, bool, type(None))) else (2, type(x).__name__))))) # Sort list elements for stability
             except (ValueError, TypeError) as e:
                 logger.warning(f"Could not process list parameter '{k}' value {repr(v)}: {e}. Converting to tuple directly.")
                 items.append((k, tuple(v))) # Basic tuple conversion on error
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
def run_backtest_for_params(params: Dict[str, Any]
                            ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    global worker_data, worker_shared_args, WORKER_STRATEGY_MAP # Access globals
    logger = logging.getLogger(__name__) # Get logger configured by initializer
    
    # +++ ADD DEBUG LOGGING FOR PARAMS ID +++
    logger.debug(f"Worker {multiprocessing.current_process().pid} ENTRY - Params object ID: {id(params)}")
    # +++ END DEBUG LOGGING +++
    
    # Check if globals were initialized correctly
    if worker_data is None or not worker_shared_args or not WORKER_STRATEGY_MAP:
        logger.error(f"Worker {multiprocessing.current_process().pid} found uninitialized globals! Skipping task.")
        return None, None

    # Extract necessary info from shared_args (now global)
    symbol = worker_shared_args["symbol"]
    trade_units = worker_shared_args["units"]
    optimize_metric = worker_shared_args["optimize_metric"]
    commission_bps = worker_shared_args["commission_bps"]
    data_start_trim = worker_shared_args["data_start_trim"]
    data_end_trim = worker_shared_args["data_end_trim"]
    apply_atr_filter = worker_shared_args["apply_atr_filter"]
    atr_filter_period = worker_shared_args["atr_filter_period"]
    atr_filter_multiplier = worker_shared_args["atr_filter_multiplier"]
    atr_filter_sma_period = worker_shared_args["atr_filter_sma_period"]
    apply_seasonality_filter = worker_shared_args["apply_seasonality_filter"]
    allowed_trading_hours_utc = worker_shared_args["allowed_trading_hours_utc"]
    strategy_short_name = worker_shared_args["strategy_short_name"]


    # Log the start of the backtest *using the passed params*
    logger.info(f"Running backtest for: {strategy_short_name} on {symbol}")
    # *** UNCOMMENT the complex INFO log for params content for now ***
    logger.info(f"Params: {params}, SL: {params.get('stop_loss')}, TP: {params.get('take_profit')}, TSL: {params.get('trailing_stop_loss')}, Comm: {commission_bps} bps")
    logger.info(f"ATR Filter: {apply_atr_filter} (P={atr_filter_period}, M={atr_filter_multiplier}, SMA={atr_filter_sma_period})")
    logger.info(f"Seasonality Filter: {apply_seasonality_filter} (Hrs={allowed_trading_hours_utc}, Syms=)") # Assuming syms is not needed per-worker

    strategy_class = WORKER_STRATEGY_MAP.get(strategy_short_name)
    if not strategy_class:
        logger.error(f"Strategy short name '{strategy_short_name}' not found in worker's STRATEGY_MAP.")
        return None, None

    # --- Prepare strategy arguments ---
    # Get the __init__ signature
    sig = inspect.signature(strategy_class.__init__)
    valid_init_params = {p for p in sig.parameters if p != 'self'}

    # Combine shared args and specific params, filtering for valid __init__ args
    # Make a copy to avoid modifying the passed 'params' dict
    current_run_params = params.copy()

    # Filter params based on the strategy's __init__ signature
    strategy_init_args = {
        k: v for k, v in current_run_params.items()
        if k in valid_init_params
    }

    # <<< START TYPE CORRECTION >>>
    # Correct types for specific strategy parameters before instantiation
    strategy_class_name = strategy_class.__name__ # Get name for checks
    if strategy_class_name == "LongShortStrategy":
         # Ensure period is int
         if 'trend_filter_period' in strategy_init_args and strategy_init_args['trend_filter_period'] is not None:
              try:
                   original_value = strategy_init_args['trend_filter_period']
                   strategy_init_args['trend_filter_period'] = int(original_value)
              except (ValueError, TypeError) as e:
                   logger.warning(f"Could not convert trend_filter_period '{original_value}' to int for {strategy_class_name}: {e}. Setting to None.")
                   strategy_init_args['trend_filter_period'] = None # Or raise error
         # Ensure use_ema is bool
         if 'trend_filter_use_ema' in strategy_init_args and strategy_init_args['trend_filter_use_ema'] is not None:
              try:
                   original_value = strategy_init_args['trend_filter_use_ema']
                   if isinstance(original_value, str):
                        strategy_init_args['trend_filter_use_ema'] = original_value.lower() in ['true', '1', 't', 'y', 'yes']
                   else:
                        strategy_init_args['trend_filter_use_ema'] = bool(original_value)
              except Exception as e:
                   logger.warning(f"Could not convert trend_filter_use_ema '{original_value}' to bool for {strategy_class_name}: {e}. Setting to False.")
                   strategy_init_args['trend_filter_use_ema'] = False # Default
    # Add similar blocks for other strategies and their specific type needs

    # <<< END TYPE CORRECTION >>>


    # Create strategy instance
    try:
        strategy_instance = strategy_class(**strategy_init_args)
    except Exception as e:
        logger.error(f"Error instantiating {strategy_class.__name__} with args {strategy_init_args}: {e}", exc_info=True)
        return None, None # Return None if strategy fails to initialize


    # --- Prepare parameters for run_backtest, extracting from params dict ---
    # Use .get() to safely access potential keys, falling back to None
    sl_val = params.get("stop_loss_pct")
    tp_val = params.get("take_profit_pct")
    tsl_val = params.get("trailing_stop_loss_pct")
    apply_atr_grid = params.get("apply_atr_filter") # Name needs to match YAML grid key
    atr_period_grid = params.get("atr_filter_period")
    # Use the key from the YAML file ('atr_filter_threshold')
    atr_multiplier_grid = params.get("atr_filter_threshold") 
    atr_sma_grid = params.get("atr_filter_sma_period")
    apply_seasonality_grid = params.get("apply_seasonality") # Name needs to match YAML grid key
    # Construct trading hours string from start/end hours in grid
    start_hour = params.get("seasonality_start_hour")
    end_hour = params.get("seasonality_end_hour")
    trading_hours_grid = f"{start_hour}-{end_hour}" if start_hour is not None and end_hour is not None else None
    
    seasonality_symbols_grid = params.get("apply_seasonality_to_symbols") # Name needs to match YAML grid key


    # Run the backtest
    try:
        result_metrics = run_backtest(
            data=worker_data.copy(), # Pass a copy to avoid potential mutation issues?
            strategy=strategy_instance,
            symbol=symbol,
            # Pass global defaults from shared args
            initial_balance=worker_shared_args.get("initial_balance", 10000.0),
            commission_bps=commission_bps,
            units=trade_units,
            apply_atr_filter=apply_atr_filter, # Global default
            atr_filter_period=atr_filter_period, # Global default
            atr_filter_multiplier=atr_filter_multiplier, # Global default
            atr_filter_sma_period=atr_filter_sma_period, # Global default
            apply_seasonality_filter=apply_seasonality_filter, # Global default
            allowed_trading_hours_utc=allowed_trading_hours_utc, # Global default
            apply_seasonality_to_symbols=worker_shared_args.get("apply_seasonality_to_symbols"), # Global default
            # Pass grid-specific values (will be None if not in params dict)
            sl_from_grid=sl_val,
            tp_from_grid=tp_val,
            tsl_from_grid=tsl_val,
            apply_atr_from_grid=apply_atr_grid,
            atr_period_from_grid=atr_period_grid,
            atr_multiplier_from_grid=atr_multiplier_grid,
            atr_sma_from_grid=atr_sma_grid,
            apply_seasonality_from_grid=apply_seasonality_grid,
            trading_hours_from_grid=trading_hours_grid,
            seasonality_symbols_from_grid=seasonality_symbols_grid,
        )
        # NOTE: run_backtest itself logs completion details.
        # Worker only needs to log if the function *call* fails or returns None.
        if not result_metrics:
             logger.warning(f"Worker {multiprocessing.current_process().pid}: run_backtest function returned None or empty metrics for params: {params}")

        # Return the original params (passed into function) and the results
        return params, result_metrics

    except Exception as e:
        logger.error(f"Exception during backtest run for params {params}: {e}", exc_info=True)
        return params, None # Return params even on failure, but None for results


def optimize_strategy(
    strategy_short_name: str,
    config_path: str,
    shared_worker_args: Dict,
    optimization_metric: str = "cumulative_profit",
    num_processes: int = DEFAULT_OPT_PROCESSES,
    log_queue: Optional[multiprocessing.Queue] = None # Argument needed
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Optimizes a single strategy for a given symbol using grid search with multiprocessing,
    avoiding redundant backtests for duplicate parameter sets.
    """
    # Get logger for this function (configured by main process)
    logger = logging.getLogger(__name__)
    symbol = shared_worker_args["symbol"]
    logger.info(
        f"Starting optimization for Strategy: {strategy_short_name}, Symbol: {symbol}"
    )
    strategy_class = STRATEGY_MAP.get(strategy_short_name)
    if not strategy_class:
        logger.error(f"Strategy name '{strategy_short_name}' not found in STRATEGY_MAP.")
        return None, None, []

    strategy_class_name = strategy_class.__name__

    try:
        # --- Load and Process Parameter Grid --- 
        config_file = Path(config_path)
        logger.info(f"Loading param grid from: {config_file}")
        if not config_file.is_file():
            raise FileNotFoundError(f"Parameter config file not found: {config_path}")

        try:
            with open(config_file, "r") as f:
                full_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error reading config file {config_path}: {e}")

        if not isinstance(full_config, dict) or symbol not in full_config:
            raise ValueError(f"Symbol '{symbol}' not found in config file: {config_path}")

        symbol_config = full_config[symbol]
        if not isinstance(symbol_config, dict) or strategy_class_name not in symbol_config:
            raise ValueError(
                f"Strategy '{strategy_class_name}' not found for symbol '{symbol}' in {config_path}"
            )

        loaded_grid = symbol_config[strategy_class_name]
        if not isinstance(loaded_grid, dict):
            raise ValueError(
                f"Parameter grid for {symbol}/{strategy_class_name} is not a dictionary."
            )

        # Process individual parameter lists (normalize 'None', dedupe values within list, sort)
        param_grid: Dict[str, List[Any]] = {}
        for key, value in loaded_grid.items():
            if not isinstance(value, list):
                logger.error(
                    f"Invalid format for parameter '{key}' in {symbol}/{strategy_class_name}. Expected a list, got {type(value)}."
                )
                raise ValueError(
                    f"Invalid format for parameter '{key}' in {symbol}/{strategy_class_name}. Expected a list, got {type(value)}."
                )

            processed_values_dict = {}
            for item in value:
                processed_item = item
                key_for_dedupe = item
                if isinstance(item, str) and item.lower() == 'none':
                    processed_item = None
                    key_for_dedupe = None
                elif isinstance(item, (int, float)):
                    key_for_dedupe = float(item)

                if key_for_dedupe not in processed_values_dict:
                    processed_values_dict[key_for_dedupe] = processed_item

            unique_values_list = list(processed_values_dict.values())

            try:
                def sort_key(x):
                    if x is None: return (0, None)
                    if isinstance(x, bool): return (1, x)
                    if isinstance(x, (int, float)): return (2, x)
                    if isinstance(x, str): return (3, x)
                    return (4, str(x))
                sorted_unique_values = sorted(unique_values_list, key=sort_key)
                param_grid[key] = sorted_unique_values
                logger.debug(f"Processed param '{key}': Original={value}, UniqueSorted={sorted_unique_values}")
            except TypeError as e:
                logger.warning(
                    f"Sorting failed for parameter '{key}' values ({unique_values_list}): {e}. Using original order of unique items."
                )
                param_grid[key] = unique_values_list
        # --- End Load and Process Parameter Grid ---

    except (FileNotFoundError, ValueError) as e:
        logger.error(
            f"Failed to load param grid for {strategy_class_name} on {symbol}: {e}"
        )
        return None, None, []

    # --- Calculate Total Possible Combinations (Before Deduplication) ---
    param_names = list(param_grid.keys())
    param_value_lists = list(param_grid.values())
    total_possible_combinations = 1
    if param_value_lists:
         try:
             for value_list in param_value_lists:
                 total_possible_combinations *= len(value_list)
         except Exception as e:
             logger.error(f"Error calculating total combinations: {e}", exc_info=True)
             total_possible_combinations = 0 # Indicate calculation error
    else:
        total_possible_combinations = 0
    logger.info(f"Total possible parameter combinations (before deduplication): {total_possible_combinations}")

    if total_possible_combinations == 0:
        logger.warning(f"No valid parameter combinations generated for {strategy_class_name} on {symbol}. Skipping.")
        return None, None, []

    # --- Generate Unique Parameter Combinations and Deduplicate On-the-Fly --- 
    unique_rep_to_params_map = {} # Stores rep -> params_dict for unique combinations
    combination_iterator = itertools.product(*param_value_lists)
    generated_count = 0

    logger.info("Generating and deduplicating parameter combinations...")
    # Iterate through all possible combinations
    for i, combination_values in enumerate(combination_iterator):
        generated_count += 1
        # Construct the dictionary for this specific combination
        params = dict(zip(param_names, combination_values))

        # Special handling for LongShortStrategy tuple params (if needed)
        if strategy_class_name == "LongShortStrategy":
            rt_low = params.pop("return_thresh_low", None)
            rt_high = params.pop("return_thresh_high", None)
            vt_low = params.pop("volume_thresh_low", None)
            vt_high = params.pop("volume_thresh_high", None)
            if rt_low is not None and rt_high is not None:
                params["return_thresh"] = (rt_low, rt_high)
            if vt_low is not None and vt_high is not None:
                params["volume_thresh"] = (vt_low, vt_high)

        # Create the hashable representation for deduplication
        rep = params_to_tuple_rep(params, precision=8) # Use desired precision

        # --- DEBUG LOGGING --- 
        # Log the params dict and its representation periodically
        log_interval = max(1, total_possible_combinations // 100) # Log ~100 times
        if i < 20 or i % log_interval == 0:
             logger.debug(f"Dedupe Check [{i}]: Params={repr(params)}, Rep={rep}")
        # --- END DEBUG LOGGING ---

        # Store the rep and params dict if the rep hasn't been seen
        # setdefault returns the existing value if key exists, otherwise sets the value and returns it
        existing = unique_rep_to_params_map.setdefault(rep, params)

        # Optional: Log when a duplicate is found (can be very verbose)
        # if existing is not params: # Check if setdefault added a new item or found existing
        #     logger.debug(f"Duplicate rep encountered: {rep}")

    # --- End Deduplication ---

    # Extract the unique parameter dictionaries from the map
    unique_params_for_backtest = list(unique_rep_to_params_map.values())
    unique_combinations_count = len(unique_params_for_backtest)
    duplicate_count = generated_count - unique_combinations_count

    logger.info(f"Generated {generated_count} total combinations.")
    logger.info(f"Deduplication complete: {unique_combinations_count} unique combinations identified for backtesting.")
    if duplicate_count > 0:
         logger.info(f"  ({duplicate_count} duplicate combinations ignored)")

    # DEBUG: Log a sample of the unique list before submitting jobs
    if unique_params_for_backtest:
         logger.debug(f"Sample of unique_params_for_backtest[0]: {unique_params_for_backtest[0]}")
         if len(unique_params_for_backtest) > 1:
              logger.debug(f"Sample of unique_params_for_backtest[1]: {unique_params_for_backtest[1]}")

    if unique_combinations_count == 0 and total_possible_combinations > 0:
        logger.error("Error: All combinations were considered duplicates. Check params_to_tuple_rep or generation logic.")
        return None, None, [] # Avoid proceeding with no jobs

    # Determine number of processes to use
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


    # --- Prepare arguments ONLY for unique backtests --- 
    # worker_args_list = [] # NO LONGER NEEDED
    # # Iterate through the *truly unique* parameter dictionaries
    # for params in unique_params_for_backtest:
    #     current_args = shared_worker_args.copy()
    #     current_args["params"] = params
    #     current_args["strategy_short_name"] = strategy_short_name
    #     worker_args_list.append(current_args)

    # +++ PRE-POOL DEBUG LOGGING +++
    # logger.info(f"Prepared {len(worker_args_list)} unique arguments for the pool.")
    # if len(worker_args_list) > 0:
    #     logger.debug(f"  ID of first arg dict: {id(worker_args_list[0])}")
    #     if len(worker_args_list) > 1:
    #         logger.debug(f"  ID of second arg dict: {id(worker_args_list[1])}")
    #         logger.debug(f"  First arg content: {worker_args_list[0]}")
    #         logger.debug(f"  Second arg content: {worker_args_list[1]}")
    # +++ END PRE-POOL DEBUG LOGGING +++

    # --- Run Backtests for Unique Combinations in Parallel ---
    unique_results_list = []
    logger.info(f"Starting parallel backtesting for {unique_combinations_count} unique combinations using {num_processes} processes...")
    
    # --- Define arguments for the initializer --- 
    # We need to pass the parts of shared_worker_args that DON'T include the large data
    # (Data will be loaded by initializer if we choose that route, or passed if small enough)
    # For now, let's pass everything needed besides data.
    init_args_for_worker = shared_worker_args.copy()
    init_args_for_worker['strategy_short_name'] = strategy_short_name # Add strategy name
    # Add any other small, constant args workers need
    # Ensure data is NOT in here if it's large and loaded by initializer
    # If data IS passed here, the initializer needs to handle it

    # *** Pass initializer correctly ***
    pool_initializer_func = None
    initializer_args = None
    if log_queue:
         # The initializer needs the queue AND the shared args (without data)
         pool_initializer_func = pool_worker_initializer_with_data
         initializer_args = (log_queue, init_args_for_worker)

    try:
        # Pass the initializer and its args to the Pool
        with Pool(processes=num_processes, 
                  initializer=pool_initializer_func, 
                  initargs=initializer_args) as pool:
            
            # *** Replace imap_unordered with map ***
            # logger.info("Submitting tasks to pool using imap_unordered (without tqdm)...")
            # imap_results = pool.imap_unordered(run_backtest_for_params, unique_params_for_backtest)
            # unique_results_list = list(imap_results) # Collect results directly
            
            logger.info("Submitting tasks to pool using map...")
            # pool.map takes the function and the iterable directly
            unique_results_list = pool.map(run_backtest_for_params, unique_params_for_backtest)
            logger.info(f"Collected {len(unique_results_list)} results from pool using map.")
            
            # Note: pool.map blocks until all tasks complete, so close/join is slightly redundant here
            # but doesn't hurt.
        logger.info("Parallel backtesting finished.")
    except Exception as e:
        logger.exception(f"Error during parallel backtesting: {e}", exc_info=True)
        return None, None, []

    # --- Process Unique Results and Find Best ---
    all_results_details = [] # Store details for each unique run
    best_params: Optional[Dict[str, Any]] = None
    minimize_metric = optimization_metric in ["max_drawdown"] # Add other minimizing metrics if needed
    best_metric_value: Optional[float] = float('inf') if minimize_metric else -float('inf')
    best_metrics_dict: Optional[Dict[str, Any]] = None # Store the full metrics dict for the best result

    processed_unique_count = 0
    successful_unique_count = 0

    logger.info(f"Processing {len(unique_results_list)} unique results returned by workers...")

    # Iterate directly over the list of unique results from the pool
    for result_params, result_metrics in unique_results_list:
        processed_unique_count += 1

        if result_params is not None and result_metrics is not None:
            successful_unique_count += 1
            # Combine params and results into a single dictionary for saving details of this unique run
            detail_entry = {f"param_{k}": v for k,v in result_params.items()}
            detail_entry.update({f"result_{k}": v for k,v in result_metrics.items()})
            all_results_details.append(detail_entry)

            # --- Update Best Result (Processing each unique result once) ---
            current_metric_value = result_metrics.get(optimization_metric)
            if current_metric_value is not None:
                is_initial_best = best_metric_value == float('inf') or best_metric_value == -float('inf')
                is_better = False
                if pd.notna(current_metric_value):
                    if is_initial_best:
                        is_better = True
                    elif minimize_metric and current_metric_value < best_metric_value:
                        is_better = True
                    elif not minimize_metric and current_metric_value > best_metric_value:
                        is_better = True
                else:
                    logger.debug(f"Metric '{optimization_metric}' has invalid value ({current_metric_value}) for params: {result_params}. Cannot compare.")

                if is_better:
                    best_metric_value = current_metric_value
                    best_params = result_params # Store the params dict
                    best_metrics_dict = result_metrics # Store the full metrics dict
                    # Log when a *new* best is found (can keep this debug log)
                    logger.debug(f"New best {optimization_metric} found: {best_metric_value:.4f} with params: {adjust_params_for_printing(best_params, strategy_class_name)}")
            # --- End Update Best Result ---
        else:
             # Log workers that returned None (error during backtest)
             # We don't have the params that failed easily here unless we modify run_backtest_for_params return value on error
             logger.warning(f"A backtest worker task completed but returned None (likely an error during execution). Skipping result.")

    # Final Summary Log
    logger.info(f"Finished processing results for {strategy_class_name} on {symbol}.")
    logger.info(f"  Processed {processed_unique_count} unique worker results.")
    logger.info(f"  Found {successful_unique_count} successful backtest results.")


    if best_params is None:
        logger.warning(f"No best parameters found for {strategy_class_name} on {symbol}. No successful backtests or specified metric '{optimization_metric}' not found?")
        # Return empty best_metrics if no best_params found
        return None, None, all_results_details
    
    # Ensure best_metric_value is formatted correctly for the final log message
    if best_metric_value is not None and best_metric_value not in [float('inf'), -float('inf')]:
         final_best_metric_value_str = f"{best_metric_value:.4f}"
    else:
         final_best_metric_value_str = "N/A" # Should not happen if best_params is not None


    logger.info(
        f"Best parameters found for {strategy_class_name} on {symbol}: "
        f"{adjust_params_for_printing(best_params, strategy_class_name)} "
        f"with {optimization_metric} = {final_best_metric_value_str}"
    )

    # Use the already stored best_metrics_dict
    if best_metrics_dict is None and best_params is not None: # This condition might occur if the best metric value was inf/-inf
        logger.warning("Could not retrieve the full metrics dictionary for the determined best parameters.")


    # Return the best params, the corresponding full metrics dict, and the list of unique result details
    return best_params, best_metrics_dict, all_results_details


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

    # --- Filter Arguments (Passed from Makefile) ---
    parser.add_argument(
        "--apply-atr-filter",
        action="store_true",
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
        action="store_true",
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

    # --- Centralized Logging Configuration --- Simplified ---
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    log_queue = multiprocessing.Queue(-1)

    # --- Configure Handlers for the Listener ---
    log_formatter = logging.Formatter(
        '%(asctime)s - %(processName)s - %(levelname)s - [%(name)s] - %(message)s'
    )
    # Use stderr to avoid interfering with tqdm progress bar
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(log_formatter)
    # Add other handlers like FileHandler here if needed

    # --- Configure the Root Logger (in the main process) ---
    root_logger = logging.getLogger() 
    # Clear any existing handlers from the root logger (important!)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    # The root logger *itself* doesn't need handlers; the listener uses them.

    # --- Start the QueueListener ---
    # Pass the queue and the handlers that will process the logs
    listener = QueueListener(log_queue, stream_handler) # Add file_handler etc. here
    listener.start()

    # Get logger instance for use in main()
    logger = logging.getLogger(__name__) 
    logger.info(f"Logging configured. Level: {args.log.upper()}. Listener started.")
    # --- End Logging Configuration ---

    try:
        logger.info(f"Starting optimization run with args: {args}")
        # ... (rest of main function: validate args, load data, pre-calculate, slice data)
        if args.apply_seasonality_filter and not args.allowed_trading_hours_utc:
            raise ValueError(
                "--allowed-trading-hours-utc must be set when --apply-seasonality-filter is enabled."
            )

        full_data = load_csv_data(str(args.file), symbol=args.symbol)
        full_data.sort_index(inplace=True)
        logger.info(
            f"Loaded full dataset: {len(full_data)} rows from {full_data.index.min()} to {full_data.index.max()}"
        )

        if args.apply_atr_filter:
            full_data["atr"] = calculate_atr(full_data, period=args.atr_filter_period)
            if args.atr_filter_sma_period > 0:
                full_data["atr_sma"] = (
                    full_data["atr"].rolling(window=args.atr_filter_sma_period).mean()
                )
            logger.info(
                f"Pre-calculated ATR (period={args.atr_filter_period}, sma={args.atr_filter_sma_period}) for full dataset."
            )

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
                raise ValueError(f"Failed to slice data for optimization: {e}")

        logger.info(
            f"Using data slice for optimization: {len(opt_data)} rows ({slice_info})"
        )

        shared_worker_args = {
            "symbol": args.symbol,
            "data": opt_data,
            "units": args.units,
            "commission_bps": args.commission,
            "initial_balance": args.balance,
            "apply_atr_filter": args.apply_atr_filter,
            "atr_filter_period": args.atr_filter_period,
            "atr_filter_multiplier": args.atr_filter_multiplier,
            "atr_filter_sma_period": args.atr_filter_sma_period,
            "apply_seasonality_filter": args.apply_seasonality_filter,
            "allowed_trading_hours_utc": args.allowed_trading_hours_utc,
            "apply_seasonality_to_symbols": args.apply_seasonality_to_symbols,
            "data_start_trim": args.opt_start,
            "data_end_trim": args.opt_end,
            "optimize_metric": args.metric,
        }

        best_params, best_result_dict, all_results_list = optimize_strategy(
            strategy_short_name=args.strategy,
            config_path=args.config,
            shared_worker_args=shared_worker_args,
            optimization_metric=args.metric,
            num_processes=args.processes,
            log_queue=log_queue # Pass the queue
        )

        # --- Save Best Parameters --- (Logic unchanged)
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

        # --- Save Detailed Results --- (Logic unchanged, uses all_results_list)
        if args.save_details:
            if all_results_list:
                try:
                    if args.details_file is None:
                        details_dir = Path("results/optimization")
                        details_dir.mkdir(parents=True, exist_ok=True)
                        filename_ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                        base_name = f"{args.strategy}_{args.symbol}"
                        filter_suffix = ""
                        if args.apply_atr_filter:
                            filter_suffix += f"_atr{args.atr_filter_period}x{args.atr_filter_multiplier:.1f}sma{args.atr_filter_sma_period}"
                        if args.apply_seasonality_filter:
                            filter_suffix += f"_season{args.allowed_trading_hours_utc.replace('-','')}"
                            if args.apply_seasonality_to_symbols:
                                syms = args.apply_seasonality_to_symbols.split(",")
                                filter_suffix += f"_{''.join(s[:3] for s in syms)}"
                        filter_suffix = sanitize_filename(filter_suffix)
                        filename = f"{base_name}{filter_suffix}_optimize_details_{filename_ts}.csv"
                        details_path = details_dir / filename
                    else:
                        details_path = Path(args.details_file)
                        details_path.parent.mkdir(parents=True, exist_ok=True)

                    # ** Adjusted Key Prefixes for DataFrame **
                    details_df = pd.DataFrame(all_results_list) # Assumes keys are already prefixed in optimize_strategy
                    
                    # Ensure correct metric key with prefix
                    metric_col = f"result_{args.metric}"
                    if metric_col in details_df.columns:
                        ascending_sort = args.metric == "max_drawdown"
                        details_df.sort_values(
                            by=metric_col, ascending=ascending_sort, inplace=True
                        )
                    else:
                        logger.warning(f"Metric column '{metric_col}' not found for sorting details CSV.")

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
        # --- Cleanly Shutdown Logging ---
        logger.info("--- Optimization script finishing. Shutting down logger... ---")
        log_queue.put_nowait(None)
        # Stop the listener
        listener.stop()
        logger.info("Logging listener stopped.")


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
