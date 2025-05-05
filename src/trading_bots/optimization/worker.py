import logging
import multiprocessing
from typing import Dict, Any, Tuple, Optional
import inspect
import os
import time
import numpy as np
import pandas as pd

# Import necessary components from other modules within the package
from ..config_models import BacktestRunConfig, ValidationError
from ..backtest import run_backtest
from .multiprocessing_utils import worker_data, worker_shared_args, WORKER_STRATEGY_MAP


# --- Worker function for parallel backtesting ---
def run_backtest_for_params(
    params: Dict[str, Any],
    symbol: str,
    run_config_dict: Dict[str, Any],
    data: pd.DataFrame,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Worker function to run a single backtest configuration."""
    worker_logger = logging.getLogger(__name__)
    worker_logger.debug(
        f"Worker {os.getpid()}: Starting backtest for {symbol} with params {params}"
    )
    start_time = time.time()

    # Check if globals were initialized correctly
    if worker_data is None or not worker_shared_args or not WORKER_STRATEGY_MAP:
        worker_logger.error(
            f"Worker {os.getpid()} found uninitialized globals! Skipping task."
        )
        # Return the params that failed, but None for metrics
        return params, None

    # --- Extract Base Config from Shared Args & Merge Grid Params --- #
    backtest_config = None
    try:
        strategy_short_name = worker_shared_args["strategy_short_name"]

        # Start with shared/fixed values from the initializer
        run_config_data = {
            "symbol": worker_shared_args["symbol"],
            "initial_balance": worker_shared_args.get("initial_balance", 10000.0),
            "commission_bps": worker_shared_args.get("commission_bps", 0.0),
            "units": worker_shared_args["units"],
            "strategy_short_name": strategy_short_name,
            # Global filter flags (defaults)
            "apply_atr_filter": worker_shared_args.get("apply_atr_filter", False),
            "apply_seasonality_filter": worker_shared_args.get(
                "apply_seasonality_filter", False
            ),
            # Default filter parameters
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
            worker_logger.error(
                f"Strategy {strategy_short_name} not found in worker map."
            )
            return params, None
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
        if "apply_atr_filter" in params:
            run_config_data["apply_atr_filter"] = params["apply_atr_filter"]
        if "apply_seasonality" in params:
            run_config_data["apply_seasonality_filter"] = params["apply_seasonality"]

        # Update specific filter values if present in grid *and corresponding apply flag is true*
        if run_config_data["apply_atr_filter"]:
            if "atr_filter_period" in params:
                run_config_data["atr_filter_period"] = params["atr_filter_period"]
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
            start_hour = params.get("seasonality_start_hour")
            end_hour = params.get("seasonality_end_hour")
            if start_hour is not None and end_hour is not None:
                run_config_data["allowed_trading_hours_utc"] = (
                    f"{start_hour}-{end_hour}"
                )
            if "apply_seasonality_to_symbols" in params:
                run_config_data["apply_seasonality_to_symbols"] = params[
                    "apply_seasonality_to_symbols"
                ]

        # --- Validate and Create Config Object --- #
        try:
            backtest_config = BacktestRunConfig(**run_config_data)
        except ValidationError as e:
            worker_logger.error(
                f"Worker {os.getpid()}: Failed to validate BacktestRunConfig for params {params}:\n{e}"
            )
            return params, {
                "error": f"Failed to validate BacktestRunConfig for params {params}:\n{e}",
                "final_balance": run_config_data.get("initial_balance", 0),
            }

    except Exception as e:
        worker_logger.error(
            f"Worker {os.getpid()}: Error preparing BacktestRunConfig in worker for params {params}: {e}",
            exc_info=True,
        )
        return params, {
            "error": f"Error preparing BacktestRunConfig: {e}",
            "final_balance": run_config_data.get("initial_balance", 0),
        }

    # Ensure backtest_config was successfully created
    if backtest_config is None:
        worker_logger.error(
            f"Worker {os.getpid()}: backtest_config preparation failed. Cannot run backtest."
        )
        return params, None

    # --- Run the backtest using the imported function --- #
    try:
        # run_backtest now returns a tuple:
        # (final_value, max_drawdown, trade_analysis, all_analyzers, plot_object)
        results_tuple = run_backtest(data=data, config=backtest_config)

        # --- Process the results tuple --- #
        if (
            results_tuple is None
        ):  # Should not happen with new error handling, but check
            worker_logger.error(
                f"Worker {os.getpid()}: run_backtest returned None unexpectedly for params {params}"
            )
            return params, {
                "error": "run_backtest returned None",
                "final_balance": backtest_config.initial_balance,
            }

        final_value, max_drawdown_pct, trade_analysis, all_analyzers_dict, _ = (
            results_tuple
        )
        # We don't need the plot_object here

        # --- Construct the results dictionary expected by the optimizer --- #
        # This should match the keys used in optimize.py for sorting/comparison
        results_dict = {
            "final_balance": final_value,
            "max_drawdown_pct": max_drawdown_pct,
            # Extract other primary metrics directly if available in all_analyzers_dict
            "cumulative_profit": all_analyzers_dict.get("returns", {}).get("rtot", 0.0)
            * 100,  # Example
            "sharpe_ratio": all_analyzers_dict.get("sharpe", {}).get(
                "sharperatio", 0.0
            ),
            "profit_factor": all_analyzers_dict.get("tradeanalyzer", {})
            .get("pnl", {})
            .get("profitfactor", 0.0),
            "total_trades": all_analyzers_dict.get("tradeanalyzer", {})
            .get("total", {})
            .get("closed", 0),
            "win_rate": (
                (
                    all_analyzers_dict.get("tradeanalyzer", {})
                    .get("won", {})
                    .get("total", 0)
                    / all_analyzers_dict.get("tradeanalyzer", {})
                    .get("total", {})
                    .get("closed", 1)
                    * 100
                )
                if all_analyzers_dict.get("tradeanalyzer", {})
                .get("total", {})
                .get("closed", 0)
                > 0
                else 0
            ),
            "sqn": all_analyzers_dict.get("sqn", {}).get("sqn", 0.0),
            # Add other metrics as needed
        }
        # Clean potential NaN/Inf values
        for k, v in results_dict.items():
            if isinstance(v, (float, int)) and (np.isnan(v) or np.isinf(v)):
                results_dict[k] = 0.0

        duration = time.time() - start_time
        worker_logger.debug(
            f"Worker {os.getpid()}: Finished backtest for {symbol}. Duration: {duration:.2f}s. Result: {results_dict.get('final_balance', 'N/A')}"
        )
        return params, results_dict

    except Exception as e:
        worker_logger.error(
            f"Worker {os.getpid()}: Exception during run_backtest call for params {params}: {e}",
            exc_info=True,
        )
        return params, {
            "error": f"Exception during run_backtest: {e}",
            "final_balance": backtest_config.initial_balance,
        }
