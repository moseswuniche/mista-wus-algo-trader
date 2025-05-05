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
import os  # Added for path manipulation

# --- Import Optimization Utilities ---
from .optimization.multiprocessing_utils import (
    pool_worker_initializer_with_data,
    worker_data,
    worker_shared_args,
    WORKER_STRATEGY_MAP,
)
from .optimization.parameter_utils import (
    params_to_tuple_rep,
    sanitize_filename,
    _load_and_prepare_param_grid,
    _generate_combinations,
)
from .optimization.worker import run_backtest_for_params

# --- Import Pydantic Models --- << NEW >>
from .config_models import OptimizeParamsConfig, ValidationError
from .config_models import BacktestRunConfig

# Assuming strategies are accessible via this import path
from .strategies.base_strategy import BaseStrategy

# --- Strategy Imports based on new config --- #
from .strategies.ma_crossover_strategy import MovingAverageCrossoverStrategy
from .strategies.scalping import ScalpingStrategy
from .strategies.bb_reversion_strategy import BollingerBandReversionStrategy
from .strategies.momentum import MomentumStrategy
from .strategies.mean_reversion_strategy import MeanReversionStrategy
from .strategies.breakout_strategy import BreakoutStrategy
from .strategies.hybrid_strategy import HybridStrategy

# --- End Strategy Imports ---

from .technical_indicators import calculate_atr
from .data_utils import load_csv_data

# --- Import Optimization Results Utilities ---
from .optimization.results import (
    _find_overall_best_from_csv,
    adjust_params_for_printing,
    save_best_params,
    get_details_filepath,
)

# Re-add module-level logger instance
logger = logging.getLogger(__name__)

# Map strategy short names (used in args and config keys) to classes
STRATEGY_MAP = {
    # Short names should ideally match the keys used in optimize_params.yaml
    "MovingAverageCrossoverStrategy": MovingAverageCrossoverStrategy,
    "ScalpingStrategy": ScalpingStrategy,
    "BollingerBandReversionStrategy": BollingerBandReversionStrategy,
    "MomentumStrategy": MomentumStrategy,
    "MeanReversionStrategy": MeanReversionStrategy,
    "BreakoutStrategy": BreakoutStrategy,
    "HybridStrategy": HybridStrategy,
    # --- Removed old entries --- #
    # "LongShort": LongShortStrategy, # Removed
    # "MACross": MovingAverageCrossoverStrategy, # Renamed key
    # "RSIReversion": RsiMeanReversionStrategy, # Removed
    # "BBReversion": BollingerBandReversionStrategy, # Renamed key
}

# --- Constants for Optimization Logging ---
PROGRESS_LOG_INTERVAL = 50  # Log progress every N combinations
DEFAULT_OPT_PROCESSES = (
    6  # Optimized default for M3 Pro (typically 6 Performance cores)
)

# Add project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# <<< End Worker Globals >>>

# --- Remove Parquet Globals and Cleanup Function ---
# (Removed _parquet_writer_instance, _details_filepath_global, _atexit_registered)
# (Removed _cleanup_parquet_writer function)
# --- End Globals and Cleanup ---

# <<< START MOVED HELPER FUNCTIONS >>>

# --- Parameter Loading & Processing ---
# --- (Moved Here - BEFORE optimize_strategy) ---


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

    # --- Pass required args from shared_worker_args --- #
    data_for_worker = shared_worker_args.get("data")
    if data_for_worker is None:
        logger.error(
            "_run_parallel_backtests: Cannot proceed without data in shared_worker_args."
        )
        return None, None
    # Pass the base config elements needed by the worker
    run_config_dict_for_worker = shared_worker_args

    # --- Wrap the worker function with fixed arguments using partial --- #
    from functools import partial  # Import here

    worker_func = partial(
        run_backtest_for_params,
        symbol=symbol,  # Fix symbol arg
        run_config_dict=run_config_dict_for_worker,  # Fix base config dict arg
        data=data_for_worker,  # Fix data arg
    )
    # Now worker_func only expects the 'params' argument, matching imap_unordered expectation

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
        # Pass the wrapped function to imap_unordered
        imap_results = pool.imap_unordered(worker_func, unique_params_for_backtest)

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
    """Orchestrates the strategy optimization process.

    Loads configuration, generates parameter combinations, runs backtests in parallel,
    analyzes results (including resuming from previous runs), and saves the best parameters.
    """
    logger = logging.getLogger(__name__)
    completed_param_reps = completed_param_reps or set()
    source_param_grid: Optional[Dict[str, List[Any]]] = None  # Initialize

    # --- Ensure details_filepath directory exists if saving details ---
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

    # 1. Load and Prepare Grid
    try:
        config_file = Path(config_path)
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_file, "r") as f:
            full_config_data = yaml.safe_load(f)
            if not isinstance(full_config_data, dict):
                raise ValueError("Config file does not contain a dictionary.")

        # Call the imported function
        param_grid, source_param_grid_for_types = _load_and_prepare_param_grid(
            full_config=full_config_data,
            symbol=symbol,
            strategy_class_name=strategy_class_name,
            cmd_apply_atr_filter=cmd_apply_atr_filter,
            cmd_apply_seasonality_filter=cmd_apply_seasonality_filter,
        )
        # Store source_param_grid needed for CSV analysis later
        source_param_grid = source_param_grid_for_types

    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        logger.error(
            f"Failed to load or prepare param grid for {strategy_class_name} on {symbol}: {e}"
        )
        return None, None

    # 2. Generate Combinations (Handles Resume)
    logger.info("Generating parameter combinations...")
    # Call the imported function
    unique_params_for_backtest = _generate_combinations(
        param_grid=param_grid,
        strategy_class_name=strategy_class_name,
        completed_param_reps=completed_param_reps,
    )
    unique_combinations_to_run_count = len(unique_params_for_backtest)
    logger.info(
        f"Generated {unique_combinations_to_run_count} unique combinations to run."
    )

    # 3. Run Backtests (if needed)
    run_best_params: Optional[Dict] = None
    run_best_metrics_dict: Optional[Dict] = None
    if unique_combinations_to_run_count == 0:
        logger.warning(
            f"No parameter combinations left to run for {strategy_class_name} on {symbol}. Skipping backtesting."
        )
    else:
        # Determine number of processes
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
        )
        # Call the local function (it uses the imported run_backtest_for_params)
        run_best_params, run_best_metrics_dict = _run_parallel_backtests(
            unique_params_for_backtest=unique_params_for_backtest,
            num_processes=num_processes,
            log_queue=log_queue,
            shared_worker_args=shared_worker_args,
            save_details=save_details,
            details_filepath=details_filepath,
            optimization_metric=optimization_metric,
            strategy_short_name=strategy_short_name,
            symbol=symbol,
        )
        logger.info("Parallel backtest run finished.")

    # 4. Determine Overall Best Params from Full History
    overall_best_params = run_best_params
    overall_best_metrics_dict = run_best_metrics_dict
    minimize_metric = optimization_metric in ["max_drawdown"]

    # Initialize best_metric_value based on run results or defaults
    if run_best_params and run_best_metrics_dict:
        best_metric_value = run_best_metrics_dict.get(
            optimization_metric, float("inf") if minimize_metric else float("-inf")
        )
    else:
        best_metric_value = float("inf") if minimize_metric else float("-inf")

    if save_details and details_filepath:
        logger.info(
            f"Analyzing details file ({details_filepath}) for overall best parameters..."
        )
        # Call the imported function
        csv_best_params, csv_best_metrics = _find_overall_best_from_csv(
            details_filepath=details_filepath,
            optimization_metric=optimization_metric,
            source_param_grid=source_param_grid,  # Pass the stored source grid
        )

        if csv_best_params is not None and csv_best_metrics is not None:
            csv_metric_value = csv_best_metrics.get(optimization_metric)
            is_csv_better = False
            if csv_metric_value is not None and pd.notna(csv_metric_value):
                # Check if current best is still the initial default
                is_initial_best = best_metric_value == float(
                    "inf"
                ) or best_metric_value == -float("inf")
                if is_initial_best:
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
                    if pd.notna(best_metric_value) and not is_initial_best
                    else "N/A"
                )
                logger.info(
                    f"Overall best from CSV ({optimization_metric}={csv_log_val}) is better than current run best ({optimization_metric}={run_log_val}). Using CSV result."
                )
                overall_best_params = csv_best_params
                overall_best_metrics_dict = csv_best_metrics
                best_metric_value = csv_metric_value  # Update best_metric_value
            else:
                logger.info(
                    "Best result from current run (if any) is better than or equal to best found in CSV. Keeping current run result."
                )
        else:
            logger.warning(
                "Could not determine overall best from CSV analysis. Using best from current run (if any)."
            )
    else:
        logger.info(
            "Skipping analysis of details file (save_details=False or details_filepath is None)."
        )

    # 5. Final Logging & Saving
    logger.info(f"Finished processing results for {strategy_class_name} on {symbol}.")
    if overall_best_params is None or overall_best_metrics_dict is None:
        logger.warning(
            f"No best parameters found for {strategy_class_name} on {symbol}. No successful backtests or metric '{optimization_metric}' not found?"
        )
    else:
        # Format final best metric value for logging
        if pd.notna(best_metric_value) and best_metric_value not in [
            float("inf"),
            -float("inf"),
        ]:
            final_best_metric_value_str = f"{best_metric_value:.4f}"
        else:
            final_best_metric_value_str = "N/A"

        # Adjust parameters for printing before logging
        # Call the imported function
        printable_best_params_for_log = adjust_params_for_printing(
            overall_best_params, strategy_class_name
        )
        logger.info(
            f"Best parameters found for {strategy_class_name} on {symbol}: "
            f"{printable_best_params_for_log} "
            f"with {optimization_metric} = {final_best_metric_value_str}"
        )

        # --- Save Individual Best Params File --- #
        try:
            output_dir = Path("results") / "optimize"
            # Call imported sanitize_filename
            safe_strategy = sanitize_filename(strategy_short_name)
            safe_symbol = sanitize_filename(symbol)
            safe_metric = sanitize_filename(optimization_metric)
            output_filename = (
                f"{safe_metric}_best_params_{safe_symbol}_{safe_strategy}.yaml"
            )
            output_filepath = output_dir / output_filename

            # Call the imported save function
            save_best_params(
                best_params=overall_best_params,
                strategy_class_name=strategy_class_name,  # Pass full class name
                details_filepath=details_filepath,  # Pass for context if needed by save_best_params
                optimization_metric=optimization_metric,
                best_metrics=overall_best_metrics_dict,
                output_filepath=output_filepath,
                # Pass command-line flags needed by save_best_params
                cmd_apply_atr_filter=cmd_apply_atr_filter,
                cmd_apply_seasonality_filter=cmd_apply_seasonality_filter,
            )
            # save_best_params logs success/failure internally

        except Exception as e:
            # Log specific error for failure during file path generation or save call setup
            logger.error(
                f"Failed to prepare for or call save_best_params for {symbol}/{strategy_short_name}: {e}",
                exc_info=True,
            )
    # --- End Individual Save --- #

    return overall_best_params, overall_best_metrics_dict


# --- End optimize_strategy ---


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trading strategy optimization.")
    # --- Add Arguments --- < RESTORED >
    parser.add_argument(
        "--strategy",
        required=True,
        help="Full class name of the strategy class to optimize (e.g., MovingAverageCrossoverStrategy). Must match keys in config.",
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
    # --- DEPRECATED: --output-config ---
    # This argument is no longer used for output by this script.
    parser.add_argument(
        "--output-config",
        help="[DEPRECATED] Path to save the best parameters YAML file. This argument is ignored. Output is saved to results/optimize/best_params_<symbol>_<strategy>.yaml.",
    )
    # --- END DEPRECATED ---
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

    # --- Final Cleanup ---
    print(f"DEBUG: Stopping log listener.", file=sys.stderr)  # Added DEBUG
    if log_listener:  # Check if listener exists before stopping
        log_listener.stop()
    print(f"DEBUG: Script finished.", file=sys.stderr)  # Added DEBUG


if __name__ == "__main__":
    main()
