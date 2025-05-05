import pandas as pd
import numpy as np
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, cast
import logging
import datetime  # Added for filename generation
import sys
import os  # Added for directory creation

# Import from new utility module
from .reporting_utils import plot_equity_curve, generate_html_report

# Import sanitize_filename
from .optimization.parameter_utils import sanitize_filename

# Assuming strategies are accessible via this import path
from .strategies import (
    # Remove old imports
    # Strategy,
    # LongShortStrategy,
    # RsiMeanReversionStrategy,
    # Import new strategy classes
    MovingAverageCrossoverStrategy,
    ScalpingStrategy,
    BollingerBandReversionStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    HybridStrategy,
)
from .backtest import run_backtest
from .technical_indicators import calculate_atr

# Add project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# --- Import BacktestRunConfig ---
from .config_models import BacktestRunConfig, ValidationError

# Assuming load_csv_data is in data_utils or similar
# This import might need adjustment based on where load_csv_data actually lives
try:
    from .data_utils import load_csv_data
except ImportError:
    # Fallback or raise error if data_utils or load_csv_data is essential and missing
    print("Warning: data_utils or load_csv_data not found. Forward test might fail.")

    # Match the signature of the actual function, including Optional return type
    def load_csv_data(
        file_path: str, symbol: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        logger.error("load_csv_data function not found due to ImportError.")
        return None  # Return None to match signature


# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Updated Strategy Map (Align with optimize.py and backtest.py) ---
STRATEGY_CLASS_MAP = {
    "MovingAverageCrossoverStrategy": MovingAverageCrossoverStrategy,
    "ScalpingStrategy": ScalpingStrategy,
    "BollingerBandReversionStrategy": BollingerBandReversionStrategy,
    "MomentumStrategy": MomentumStrategy,
    "MeanReversionStrategy": MeanReversionStrategy,
    "BreakoutStrategy": BreakoutStrategy,
    "HybridStrategy": HybridStrategy,
}

# --- Utility Functions ---


def parse_trading_hours(hours_str: str) -> Optional[Tuple[int, int]]:
    """Parses a string like '5-17' into start and end hours."""
    if not hours_str or "-" not in hours_str:
        return None
    try:
        start, end = map(int, hours_str.split("-"))
        if 0 <= start <= 23 and 0 <= end <= 23 and start < end:
            return start, end
        else:
            logger.warning(
                f"Invalid hour range: {hours_str}. Must be 0-23 and start < end."
            )
            return None
    except ValueError:
        logger.warning(f"Could not parse trading hours: {hours_str}")
        return None


# --- Function to create results directory ---
def ensure_results_dir(base_path: str) -> Path:
    results_path = Path(base_path)
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path


# --- Main Forward Testing Function ---
def run_forward_test(
    strategy_class_name: str,  # Use full class name
    param_config_path: str,  # Path to the BEST params YAML
    symbol: str,
    data: pd.DataFrame,  # Data ALREADY potentially sliced by main()
    trade_units: float,
    commission_bps: float,
    initial_balance: float,
    # Global filter args are NOT needed here anymore if using BacktestRunConfig properly
    # Reporting args
    fwd_test_start: Optional[str] = None,
    fwd_test_end: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Runs a backtest on the forward test data slice using optimized parameters
    loaded from the specified configuration file.
    Constructs BacktestRunConfig and calls run_backtest.
    """
    logger.info(f"--- Running Forward Test for {strategy_class_name} on {symbol} --- ")
    logger.info(f"Loading optimized parameters from: {param_config_path}")

    # 1. Validate Strategy Class Name
    if strategy_class_name not in STRATEGY_CLASS_MAP:
        logger.error(f"Unknown strategy class name: {strategy_class_name}")
        return None
    # strategy_class = STRATEGY_CLASS_MAP[strategy_class_name] # Not needed here

    # 2. Load Parameters from the best_params YAML file
    param_file = Path(param_config_path)
    if not param_file.is_file():
        logger.error(f"Parameter config file not found: {param_config_path}")
        return None
    try:
        with open(param_file, "r") as f:
            loaded_data = yaml.safe_load(f)
            if not isinstance(loaded_data, dict) or "parameters" not in loaded_data:
                logger.error(
                    f"Invalid format in {param_config_path}. Expected dict with 'parameters' key."
                )
                return None

            # --- Get the core parameters dictionary ---
            best_params_all = loaded_data.get("parameters")
            if not isinstance(best_params_all, dict):
                logger.error(
                    f"'parameters' key in {param_config_path} does not contain a dictionary."
                )
                return None

            # --- Extract any metadata needed if saved separately ---
            # Example: Original filters used during optimization
            optimization_filters = loaded_data.get("optimization_filters", {})

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {param_config_path}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"Error reading or processing parameter config {param_config_path}: {e}",
            exc_info=True,
        )
        return None

    if not best_params_all:
        # Should not happen if checks above passed, but as a safeguard
        logger.error(f"Failed to load parameters dictionary from {param_config_path}.")
        return None

    logger.info(f"Loaded best parameters: {best_params_all}")

    # 3. Construct BacktestRunConfig
    try:
        # --- Define Output Paths --- #
        results_base_path = Path("results") / "forward_test"
        plots_path = results_base_path / "plots"
        trades_path = results_base_path / "trades"
        reports_path = results_base_path / "reports"

        # Create base filename elements
        safe_strategy = sanitize_filename(strategy_class_name)
        safe_symbol = sanitize_filename(symbol)
        start_str = (
            pd.to_datetime(fwd_test_start).strftime("%Y%m%d")
            if fwd_test_start
            else "start"
        )
        end_str = (
            pd.to_datetime(fwd_test_end).strftime("%Y%m%d")
            if fwd_test_end
            else "latest"
        )
        # Try to get metric prefix from param filename if possible (for better uniqueness)
        param_filename_stem = Path(param_config_path).stem
        metric_prefix = "optimized"  # Default prefix
        if "best_params" in param_filename_stem:
            parts = param_filename_stem.split("_")
            if len(parts) > 2 and parts[-2] == "params":
                potential_metric = parts[0]
                if potential_metric in [
                    "sharpe_ratio",
                    "cumulative_profit",
                    "profit_factor",
                    "max_drawdown",
                    "win_rate",
                ]:
                    metric_prefix = potential_metric

        base_filename = (
            f"fwd_{metric_prefix}_{safe_symbol}_{safe_strategy}_{start_str}_{end_str}"
        )

        # Construct full paths
        plot_file_path = plots_path / f"{base_filename}_plot.png"
        trade_list_file_path = trades_path / f"{base_filename}_trades.csv"
        report_file_path = reports_path / f"{base_filename}_report.html"

        # Ensure directories exist
        os.makedirs(plots_path, exist_ok=True)
        os.makedirs(trades_path, exist_ok=True)
        os.makedirs(reports_path, exist_ok=True)

        # --- Populate config_dict --- #
        config_dict = {
            # Core settings from function args
            "symbol": symbol,
            "initial_balance": initial_balance,
            "commission_bps": commission_bps,
            "units": trade_units,
            "strategy_class_name": strategy_class_name,
            "strategy_params": best_params_all,
            "stop_loss_pct": best_params_all.get("stop_loss_pct"),
            "take_profit_pct": best_params_all.get("take_profit_pct"),
            "time_window": best_params_all.get("time_window"),
            "liquidity_threshold": best_params_all.get("liquidity_threshold"),
            # --- Add File Paths to Config --- #
            "plot_path": str(plot_file_path.resolve()),
            "trade_list_output_path": str(trade_list_file_path.resolve()),
        }

        # --- Validate and Create Config Object --- #
        # Clean None strings just in case they exist
        cleaned_config_dict = {
            k: (None if isinstance(v, str) and v.lower() == "none" else v)
            for k, v in config_dict.items()
        }
        # Also clean within strategy_params if necessary (though base_strategy handles this now)
        if isinstance(cleaned_config_dict.get("strategy_params"), dict):
            # Ensure we don't overwrite if it wasn't a dict initially
            temp_params = cleaned_config_dict.get("strategy_params", {})
            # Ignore potential union-attr error if temp_params is somehow not iterable
            cleaned_config_dict["strategy_params"] = {
                k: (None if isinstance(v, str) and v.lower() == "none" else v)
                for k, v in temp_params.items() # type: ignore[union-attr]
            }

        logger.debug(
            f"Attempting to create BacktestRunConfig with: {cleaned_config_dict}"
        )
        # Use explicit field assignment with .get() for safety
        # And add type validation/casting before passing to Pydantic model

        # Validate/Cast required fields first
        try:
            symbol_val = str(cleaned_config_dict.get("symbol", ""))
            initial_balance_val = float(cleaned_config_dict.get("initial_balance", 0.0))
            commission_bps_val = float(cleaned_config_dict.get("commission_bps", 0.0))
            units_val = float(cleaned_config_dict.get("units", 0.0))
            strategy_short_name_val = str(
                cleaned_config_dict.get("strategy_class_name", "")
            )
            strategy_params_val = dict(cleaned_config_dict.get("strategy_params", {})) # type: ignore[arg-type]
        except (ValueError, TypeError) as e:
            logger.error(f"Type error converting required config values: {e}")
            return None  # Cannot proceed if required fields fail validation

        # Get optional fields with defaults and type safety
        stop_loss_pct_val = cleaned_config_dict.get("stop_loss_pct")
        take_profit_pct_val = cleaned_config_dict.get("take_profit_pct")
        plot_path_val = cleaned_config_dict.get("plot_path")
        trade_list_output_path_val = cleaned_config_dict.get("trade_list_output_path")

        # Explicitly cast optional values, handling None
        try:
            stop_loss_pct_casted = (
                float(stop_loss_pct_val) # type: ignore
                if stop_loss_pct_val is not None
                else None
            )
            take_profit_pct_casted = (
                float(take_profit_pct_val) # type: ignore
                if take_profit_pct_val is not None
                else None
            )
            plot_path_casted = str(plot_path_val) if plot_path_val is not None else None
            trade_list_output_path_casted = (
                str(trade_list_output_path_val)
                if trade_list_output_path_val is not None
                else None
            )

            # Filter params from best_params_all (load source already done)
            apply_atr_filter_casted = bool(
                best_params_all.get("apply_atr_filter", False)
            )
            atr_filter_period_casted = int(best_params_all.get("atr_filter_period", 14))
            atr_filter_multiplier_casted = float( # type: ignore
                best_params_all.get("atr_filter_multiplier", 1.5)
            )
            atr_filter_sma_period_casted = int(
                best_params_all.get("atr_filter_sma_period", 100)
            )
            apply_seasonality_filter_casted = bool(
                best_params_all.get("apply_seasonality_filter", False)
            )
            allowed_trading_hours_utc_val = best_params_all.get(
                "allowed_trading_hours_utc"
            )
            allowed_trading_hours_utc_casted = (
                str(allowed_trading_hours_utc_val)
                if allowed_trading_hours_utc_val is not None
                else None
            )
            apply_seasonality_to_symbols_val = best_params_all.get(
                "apply_seasonality_to_symbols"
            )
            apply_seasonality_to_symbols_casted = (
                str(apply_seasonality_to_symbols_val)
                if apply_seasonality_to_symbols_val is not None
                else None
            )

        except (ValueError, TypeError) as e:
            logger.error(f"Type error converting optional config values: {e}")
            return None

        backtest_config = BacktestRunConfig(
            symbol=symbol_val,
            initial_balance=initial_balance_val,
            commission_bps=commission_bps_val,
            units=units_val,
            strategy_short_name=strategy_short_name_val,
            strategy_params=strategy_params_val, # type: ignore[arg-type] # Ignore potential object type for dict
            stop_loss_pct=stop_loss_pct_casted, # type: ignore[arg-type] # Ignore potential object
            take_profit_pct=take_profit_pct_casted, # type: ignore[arg-type] # Ignore potential object
            # Include casted filter params
            apply_atr_filter=apply_atr_filter_casted,
            atr_filter_period=atr_filter_period_casted,
            atr_filter_multiplier=atr_filter_multiplier_casted,
            atr_filter_sma_period=atr_filter_sma_period_casted,
            apply_seasonality_filter=apply_seasonality_filter_casted,
            allowed_trading_hours_utc=allowed_trading_hours_utc_casted,
            apply_seasonality_to_symbols=apply_seasonality_to_symbols_casted,
            # Pass casted file paths
            plot_path=plot_path_casted,
            trade_list_output_path=trade_list_output_path_casted,
        ) # type: ignore[call-arg] # Ignore stubborn arg-type errors from config values
        logger.info("Successfully created BacktestRunConfig for forward test.")

    except ValidationError as e:
        logger.error(
            f"Failed to validate BacktestRunConfig for forward test with params from {param_config_path}:\n{e}"
        )
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating BacktestRunConfig: {e}", exc_info=True)
        return None

    # 4. Run Backtest with the specific config
    logger.info("Calling run_backtest with forward test data and loaded config...")
    # The run_backtest function now uses backtrader engine
    # It uses plot_path and trade_list_output_path from the config object
    backtest_result_tuple = run_backtest(
        data=data,
        config=backtest_config,
        # plot_filename argument removed
    )

    if backtest_result_tuple is None:
        logger.error("run_backtest failed during forward test.")
        return None

    final_value, max_drawdown, trade_analysis, all_analyzers, plot_object = (
        backtest_result_tuple
    )

    # 5. Generate Report
    report_metrics: Dict[str, Any] = {} # Initialize with type hint

    # --- Safely extract metrics with type checks and casts ---
    final_value_raw = final_value
    max_drawdown_raw = max_drawdown

    # --- Returns Analyzer ---
    returns_analyzer = all_analyzers.get('returns')
    cum_return_raw = None
    ann_return_raw = None
    if isinstance(returns_analyzer, dict):
        returns_analyzer_dict = cast(Dict[str, Any], returns_analyzer)
        cum_return_raw = returns_analyzer_dict.get('rtot')
        ann_return_raw = returns_analyzer_dict.get('rnorm100')
    else:
        logger.warning("Returns analyzer data is not a dictionary.")

    # --- Sharpe Analyzer ---
    sharpe_analyzer = all_analyzers.get('sharpe')
    sharpe_raw = None
    if isinstance(sharpe_analyzer, dict):
        sharpe_analyzer_dict = cast(Dict[str, Any], sharpe_analyzer)
        sharpe_raw = sharpe_analyzer_dict.get('sharperatio')
    else:
        logger.warning("Sharpe analyzer data is not a dictionary.")

    # --- Trade Analyzer ---
    trade_analyzer = all_analyzers.get('tradeanalyzer')
    total_trades_raw = None
    winning_trades_raw = None
    losing_trades_raw = None
    avg_pnl_raw = None
    avg_win_pnl_raw = None
    avg_loss_pnl_raw = None
    max_win_streak_raw = None
    max_loss_streak_raw = None
    total_won_pnl_raw = None
    total_lost_pnl_raw = None # Usually negative

    if isinstance(trade_analyzer, dict):
        trade_analyzer_dict = cast(Dict[str, Any], trade_analyzer)

        total_dict = trade_analyzer_dict.get('total')
        if isinstance(total_dict, dict):
            total_trades_raw = cast(Dict[str, Any], total_dict).get('closed') # type: ignore[union-attr]

        won_dict = trade_analyzer_dict.get('won')
        if isinstance(won_dict, dict):
            won_dict_cast = cast(Dict[str, Any], won_dict)
            winning_trades_raw = won_dict_cast.get('total')
            won_pnl_dict = won_dict_cast.get('pnl')
            if isinstance(won_pnl_dict, dict):
                 won_pnl_dict_cast = cast(Dict[str, Any], won_pnl_dict) # type: ignore[arg-type]
                 avg_win_pnl_raw = won_pnl_dict_cast.get('average')
                 total_won_pnl_raw = won_pnl_dict_cast.get('total')

        lost_dict = trade_analyzer_dict.get('lost')
        if isinstance(lost_dict, dict):
            lost_dict_cast = cast(Dict[str, Any], lost_dict)
            losing_trades_raw = lost_dict_cast.get('total')
            lost_pnl_dict = lost_dict_cast.get('pnl')
            if isinstance(lost_pnl_dict, dict):
                 lost_pnl_dict_cast = cast(Dict[str, Any], lost_pnl_dict) # type: ignore[arg-type]
                 avg_loss_pnl_raw = lost_pnl_dict_cast.get('average')
                 total_lost_pnl_raw = lost_pnl_dict_cast.get('total') # Negative

        pnl_dict = trade_analyzer_dict.get('pnl')
        if isinstance(pnl_dict, dict):
            pnl_dict_cast = cast(Dict[str, Any], pnl_dict)
            net_pnl_dict = pnl_dict_cast.get('net')
            if isinstance(net_pnl_dict, dict):
                 avg_pnl_raw = cast(Dict[str, Any], net_pnl_dict).get('average') # type: ignore[union-attr]

        streak_dict = trade_analyzer_dict.get('streak')
        if isinstance(streak_dict, dict):
            streak_dict_cast = cast(Dict[str, Any], streak_dict)
            won_streak_dict = streak_dict_cast.get('won')
            if isinstance(won_streak_dict, dict):
                max_win_streak_raw = cast(Dict[str, Any], won_streak_dict).get('longest') # type: ignore[union-attr]
            lost_streak_dict = streak_dict_cast.get('lost')
            if isinstance(lost_streak_dict, dict):
                max_loss_streak_raw = cast(Dict[str, Any], lost_streak_dict).get('longest') # type: ignore[union-attr]
    else:
        logger.warning("Trade analyzer data is not a dictionary.")

    # --- SQN Analyzer ---
    sqn_analyzer = all_analyzers.get('sqn')
    sqn_raw = None
    if isinstance(sqn_analyzer, dict):
        sqn_analyzer_dict = cast(Dict[str, Any], sqn_analyzer)
        sqn_raw = sqn_analyzer_dict.get('sqn')
    else:
        logger.warning("SQN analyzer data is not a dictionary.")

    # --- Populate report_metrics with safe casting ---
    # Wrap the whole block in try/except for safety during conversion
    try:
        # Helper function to safely cast to float
        def safe_float(value: Any) -> Optional[float]:
            if value is None: return None
            try: return float(value) # type: ignore[arg-type]
            except (ValueError, TypeError): return None

        # Helper function to safely cast to int
        def safe_int(value: Any) -> Optional[int]:
             if value is None: return None
             try: return int(float(value)) # type: ignore[arg-type]
             except (ValueError, TypeError): return None

        report_metrics['final_balance'] = safe_float(final_value_raw) or 0.0
        report_metrics['max_drawdown_pct'] = safe_float(max_drawdown_raw) or 0.0

        cum_return_float = safe_float(cum_return_raw)
        report_metrics['cumulative_return_pct'] = (cum_return_float * 100) if cum_return_float is not None else 0.0
        report_metrics['annualized_return_pct'] = safe_float(ann_return_raw) or 0.0
        report_metrics['sharpe_ratio'] = safe_float(sharpe_raw) # Can be None

        report_metrics['total_trades'] = safe_int(total_trades_raw) or 0
        report_metrics['winning_trades'] = safe_int(winning_trades_raw) or 0
        report_metrics['losing_trades'] = safe_int(losing_trades_raw) or 0

        total_trades_val = report_metrics['total_trades']
        winning_trades_val = report_metrics['winning_trades']
        report_metrics['win_rate'] = (winning_trades_val / total_trades_val * 100) if total_trades_val > 0 else 0.0

        report_metrics['average_trade_pnl'] = safe_float(avg_pnl_raw) or 0.0
        report_metrics['average_winning_trade'] = safe_float(avg_win_pnl_raw) or 0.0
        report_metrics['average_losing_trade'] = safe_float(avg_loss_pnl_raw) or 0.0

        report_metrics['max_consecutive_wins'] = safe_int(max_win_streak_raw) or 0
        report_metrics['max_consecutive_losses'] = safe_int(max_loss_streak_raw) or 0

        total_won_pnl = safe_float(total_won_pnl_raw) or 0.0
        total_lost_pnl = abs(safe_float(total_lost_pnl_raw) or 0.0) # type: ignore[arg-type] # Ignore potential None to abs()
        report_metrics["profit_factor"] = (total_won_pnl / total_lost_pnl) if total_lost_pnl > 1e-9 else float('inf') if total_won_pnl > 1e-9 else 0.0 # Avoid division by zero/near-zero

        report_metrics["sqn"] = safe_float(sqn_raw) # Can be None

    except ZeroDivisionError:
         logger.error("Division by zero error during metric calculation.", exc_info=True)
         report_metrics['win_rate'] = report_metrics.get('win_rate', 0.0)
         report_metrics['profit_factor'] = report_metrics.get('profit_factor', 0.0)

    except Exception as e:
         logger.error(f"Error processing or casting raw metrics: {e}", exc_info=True)
         return None # Exit if metrics can't be processed

    # --- Generate Report ---
    trade_list_path = backtest_config.trade_list_output_path # Get path from config
    plot_path = backtest_config.plot_path

    report_title = f"Forward Test Report: {strategy_class_name} on {symbol} ({start_str} to {end_str})"
    # Ensure plot_path and trade_list_path are Path objects or None for generate_html_report
    plot_path_obj = Path(plot_path) if plot_path and Path(plot_path).exists() else None
    trade_list_path_obj = Path(trade_list_path) if trade_list_path and Path(trade_list_path).exists() else None

    success = generate_html_report(
        metrics=report_metrics, # Pass the cleaned and typed metrics
        plot_path=plot_path_obj, # Pass the Path object or None
        report_filename=report_file_path, # Pass the Path object
        report_title=report_title,
        test_params={
            "Symbol": symbol,
            "Forward Start": fwd_test_start,
            "Forward End": fwd_test_end or "End of Data",
            "Initial Balance": initial_balance,
            "Commission BPS": commission_bps,
            "Units": trade_units,
            "Data Start": data.index.min().strftime("%Y-%m-%d %H:%M"),
            "Data End": data.index.max().strftime("%Y-%m-%d %H:%M"),
            "Parameter Config": param_config_path,
        },
        strategy_params=best_params_all, # Pass the loaded strategy params
        trade_list_path=trade_list_path_obj, # Pass the Path object or None
    )

    if success:
        logger.info(f"Generated HTML report: {report_file_path}")
    else:
        logger.warning("HTML report generation failed.")

    # --- Return dictionary with paths and core metrics ---
    results = {
        "final_balance": report_metrics.get("final_balance"),
        "max_drawdown_pct": report_metrics.get("max_drawdown_pct"),
        "total_trades": report_metrics.get("total_trades"),
        "win_rate": report_metrics.get("win_rate"),
        "sharpe_ratio": report_metrics.get("sharpe_ratio"),
        "profit_factor": report_metrics.get("profit_factor"),
        "sqn": report_metrics.get("sqn"),
        "plot_file_path": str(plot_path_obj.resolve()) if plot_path_obj else None,
        "trade_list_output_path": str(trade_list_path_obj.resolve()) if trade_list_path_obj else None,
        "report_file_path": str(report_file_path.resolve()) if report_file_path and report_file_path.exists() else None,
    }

    logger.info("Forward test run_backtest completed.")
    return results # Return metrics and paths


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Simulated Forward Test using Optimized Parameters"
    )
    # --- Core Arguments ---
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        required=True,
        choices=list(STRATEGY_CLASS_MAP.keys()),
        help="Full CLASS NAME of the strategy to test (e.g., MovingAverageCrossoverStrategy).",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading symbol (e.g., BTCUSDT) to load data and parameters for.",
    )
    parser.add_argument(
        "--param-config",
        type=str,
        required=True,
        help="Path to the YAML file containing the optimized parameters (e.g., results/optimize/best_params_SYMBOL_STRATEGY.yaml).",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,  # Make file required
        help="Path to the FULL historical data CSV file.",
    )

    # --- Test Period Arguments ---
    parser.add_argument(
        "--fwd-start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD) for the forward testing period.",
    )
    parser.add_argument(
        "--fwd-end",
        type=str,
        default=None,
        help="Optional end date (YYYY-MM-DD) for the forward testing period. If None, uses data until the end.",
    )

    # --- Backtest Parameters ---
    parser.add_argument(
        "-u", "--units", type=float, default=1.0, help="Amount/Units of asset to trade."
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=10.0,
        help="Commission fee in basis points (e.g., 10 for 0.1%). Default: 10.",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10000.0,
        help="Initial balance for the forward test backtest. Default: 10000.0",
    )
    # --- SL/TP/TSL Arguments ---
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop Loss percentage (e.g., 0.05 for 5%). Overrides strategy default if set.",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Take Profit percentage (e.g., 0.10 for 10%). Overrides strategy default if set.",
    )
    parser.add_argument(
        "--trailing-stop-loss",
        type=float,
        default=None,
        help="Trailing Stop Loss percentage (e.g., 0.02 for 2%). Overrides strategy default if set.",
    )

    # --- Filter Arguments ---
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
        default=None,  # Keep default None here, validate later
        help="Allowed trading hours in UTC (e.g., '5-17'). Required if seasonality filter is enabled.",
    )
    parser.add_argument(
        "--apply-seasonality-to-symbols",
        type=str,
        default=None,  # Keep default None here
        help="Comma-separated list of symbols to apply seasonality filter to (if empty, applies to the main symbol).",
    )

    # --- Reporting Arguments ---
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/forward_test",
        help="Directory to save analysis results (plots, reports, trades).",
    )
    parser.add_argument(
        "--plotting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable generation of plots and HTML reports.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "--optimization-metric-prefix",
        type=str,
        default=None,
        help="Optional prefix for report filenames, typically the optimization metric.",
    )

    args = parser.parse_args()

    # --- Set Log Level ---
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.info(f"Forward test script started with args: {args}")

    # --- Validate Arguments ---
    if args.apply_seasonality_filter and not args.allowed_trading_hours_utc:
        logger.error(
            "Error: --allowed-trading-hours-utc must be provided when --apply-seasonality-filter is used."
        )
        exit(1)
    if args.allowed_trading_hours_utc and not parse_trading_hours(
        args.allowed_trading_hours_utc
    ):
        logger.error(
            f"Error: Invalid format for --allowed-trading-hours-utc: {args.allowed_trading_hours_utc}. Use HH-HH format (e.g., 5-17)."
        )
        exit(1)

    # --- Load and Prepare Data ---
    try:
        # Convert Path to string for load_csv_data
        full_data = load_csv_data(str(Path(args.file)), symbol=args.symbol)

        # --- Add explicit None check --- >
        if full_data is None:
            # Error is already logged by load_csv_data
            logger.critical(f"Failed to load data from {args.file}. Exiting.")
            exit(1)
        # --- End None check --- >

        full_data.sort_index(inplace=True)
        logger.info(f"Loaded {len(full_data)} rows from {args.file}")

        # Slice data for the specified forward period
        fwd_start_dt = pd.to_datetime(args.fwd_start)
        fwd_end_dt = pd.to_datetime(args.fwd_end) if args.fwd_end else None
        forward_data_slice = full_data.loc[fwd_start_dt:fwd_end_dt].copy()

        if forward_data_slice.empty:
            logger.error(
                f"No data found for the specified forward test period ({args.fwd_start} to {args.fwd_end or 'end'}). Check data file and dates."
            )
            exit(1)
        logger.info(
            f"Sliced data for forward test: {len(forward_data_slice)} rows from {forward_data_slice.index.min()} to {forward_data_slice.index.max()}"
        )

        # Pre-calculate indicators needed for filters
        if args.apply_atr_filter:
            forward_data_slice["atr"] = calculate_atr(
                forward_data_slice, period=args.atr_filter_period
            )
            if args.atr_filter_sma_period > 0:
                forward_data_slice["atr_sma"] = (
                    forward_data_slice["atr"]
                    .rolling(window=args.atr_filter_sma_period)
                    .mean()
                )
            logger.info("Pre-calculated ATR for forward test data.")

    except FileNotFoundError:
        logger.error(f"Error: Data file not found at {args.file}")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading or processing data: {e}", exc_info=True)
        exit(1)

    # --- Run Forward Test ---
    fwd_test_output = run_forward_test(
        strategy_class_name=args.strategy,
        param_config_path=args.param_config,  # Path to best_params.yaml
        symbol=args.symbol,
        data=forward_data_slice,  # Pass the prepared slice
        trade_units=args.units,
        commission_bps=args.commission,
        initial_balance=args.balance,
        fwd_test_start=args.fwd_start,  # Pass for logging/reporting
        fwd_test_end=args.fwd_end,  # Pass for logging/reporting
    )

    if fwd_test_output is None:
        logger.error("Forward test execution failed.")
        sys.exit(1)

    # --- Save Results ---
    try:
        results_path = ensure_results_dir(args.results_dir)
        # Generate a filename base (similar to before, maybe refine)
        safe_strategy = sanitize_filename(args.strategy)
        safe_symbol = sanitize_filename(args.symbol)
        fwd_start_dt = pd.to_datetime(args.fwd_start)
        fwd_end_dt = pd.to_datetime(args.fwd_end) if args.fwd_end else None
        start_str = fwd_start_dt.strftime("%Y%m%d")
        end_str = fwd_end_dt.strftime("%Y%m%d") if fwd_end_dt else "latest"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prefix = (
            sanitize_filename(args.optimization_metric_prefix)
            if args.optimization_metric_prefix
            else None
        )

        base_filename_parts = []
        if safe_prefix:
            base_filename_parts.append(safe_prefix)
        base_filename_parts.extend(
            [safe_strategy, safe_symbol, "fwd", start_str, end_str, timestamp]
        )
        base_filename = "_".join(base_filename_parts)

        # --- Get Plot Path from results --- #
        # If plotting was disabled in run_backtest (e.g. matplotlib missing),
        # plot_file_path might be None or file might not exist.
        generated_plot_path = fwd_test_output.get("plot_file_path")
        plot_exists = generated_plot_path and Path(generated_plot_path).exists()

        # 1. Save Metrics to YAML
        metrics_filename = results_path / f"{base_filename}_metrics.yaml"
        yaml_metrics = {}
        for k, v in fwd_test_output.items():
            # Exclude plot path and other non-metric items if needed
            if k == "plot_file_path" or isinstance(v, (pd.DataFrame, pd.Series)):
                continue
            elif isinstance(v, (datetime.datetime, pd.Timestamp)):
                yaml_metrics[k] = v.isoformat()
            else:
                yaml_metrics[k] = v
        with open(metrics_filename, "w") as f:
            yaml.dump(yaml_metrics, f, default_flow_style=False)
        logger.info(f"Saved forward test metrics to: {metrics_filename}")

        # 2. Generate and Save HTML Report (if plotting enabled and plot exists)
        if args.plotting:
            if plot_exists:
                try:
                    html_report_filename = (
                        results_path / "reports" / f"{base_filename}_report.html"
                    )
                    html_report_filename.parent.mkdir(parents=True, exist_ok=True)

                    # --- Extract best_params for report --- #
                    # Need to reload from config file as run_forward_test doesn't return them anymore
                    best_params_for_report = {}
                    try:
                        param_file = Path(args.param_config)
                        with open(param_file, "r") as f:
                            loaded_data = yaml.safe_load(f)
                            best_params_for_report = loaded_data.get("parameters", {})
                    except Exception as e:
                        logger.warning(
                            f"Could not reload best params from {args.param_config} for report: {e}"
                        )

                    # --- Prepare parameters for the report --- #
                    # Get core strategy params
                    strategy_params_for_report = {
                        k: v
                        for k, v in best_params_for_report.items()
                        if k
                        not in [  # Exclude base/filter keys
                            "stop_loss_pct",
                            "take_profit_pct",
                            "time_window",
                            "liquidity_threshold",
                            # Explicitly list all filter params from BacktestRunConfig
                            "apply_atr_filter",
                            "atr_filter_period",
                            "atr_filter_multiplier",
                            "atr_filter_sma_period",
                            "apply_seasonality_filter",
                            "allowed_trading_hours_utc",
                            "apply_seasonality_to_symbols",
                        ]
                    }
                    # Clean None strings
                    strategy_params_for_report = {
                        k: (None if isinstance(v, str) and v.lower() == "none" else v)
                        for k, v in strategy_params_for_report.items()
                    }

                    # Get test parameters (use best_params for filters where available)
                    test_params_for_report = {
                        "Symbol": args.symbol,
                        "Forward Start": args.fwd_start,
                        "Forward End": args.fwd_end or "End of Data",
                        "Initial Balance": args.balance,
                        "Commission (bps)": args.commission,
                        "Units": args.units,
                        # Add filter status derived from best_params
                        "Time Window": best_params_for_report.get("time_window", "N/A"),
                        "Liquidity Threshold": best_params_for_report.get(
                            "liquidity_threshold", "N/A"
                        ),
                        "Stop Loss Pct": best_params_for_report.get(
                            "stop_loss_pct", "N/A"
                        ),
                        "Take Profit Pct": best_params_for_report.get(
                            "take_profit_pct", "N/A"
                        ),
                        # Add filter param values
                        "Apply ATR Filter": best_params_for_report.get(
                            "apply_atr_filter", "N/A"
                        ),
                        "ATR Period": best_params_for_report.get(
                            "atr_filter_period", "N/A"
                        ),
                        "ATR Multiplier": best_params_for_report.get(
                            "atr_filter_multiplier", "N/A"
                        ),
                        "ATR SMA Period": best_params_for_report.get(
                            "atr_filter_sma_period", "N/A"
                        ),
                        "Apply Seasonality": best_params_for_report.get(
                            "apply_seasonality_filter", "N/A"
                        ),
                        "Trading Hours UTC": best_params_for_report.get(
                            "allowed_trading_hours_utc", "N/A"
                        ),
                        "Seasonality Symbols": best_params_for_report.get(
                            "apply_seasonality_to_symbols", "N/A"
                        ),
                    }

                    # Remove unexpected args: config_details, optimization_plot_paths, output_filename
                    generate_html_report(
                        metrics=fwd_test_output,  # Use the prepared metrics dict
                        report_filename=html_report_filename,  # Use correct arg name
                        plot_path=generated_plot_path,  # Pass path to the backtrader plot
                        trade_list_path=fwd_test_output.get(
                            "trade_list_output_path"
                        ),  # Pass trade list path
                        report_title=f"Forward Test Report: {args.strategy} on {args.symbol}",  # Use args for title
                        test_params=test_params_for_report,
                        strategy_params=strategy_params_for_report,
                    )
                    logger.info(f"Generated HTML report: {html_report_filename}")
                except NameError:
                    logger.warning(
                        "generate_html_report function not available. Skipping HTML report."
                    )
                except Exception as e:
                    logger.error(f"Error generating HTML report: {e}", exc_info=True)
            else:
                logger.warning(
                    f"Plotting enabled, but plot file was not found or not generated: {generated_plot_path}. Skipping HTML report plot embedding."
                )
        else:
            logger.info("Plotting disabled by user.")

    except Exception as e:
        logger.error(f"Error saving results: {e}", exc_info=True)
        sys.exit(1)

    logger.info("--- Forward test script finished. ---")
