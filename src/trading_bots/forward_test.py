import pandas as pd
import numpy as np
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, cast
import logging
import datetime  # Added for filename generation
import sys

# Import from new utility module
from .reporting_utils import plot_equity_curve, generate_html_report

# Assuming strategies are accessible via this import path
from .strategies import (
    Strategy,
    LongShortStrategy,
    MovingAverageCrossoverStrategy,
    RsiMeanReversionStrategy,
    BollingerBandReversionStrategy,
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

    def load_csv_data(file_path: str, symbol: Optional[str] = None) -> pd.DataFrame:
        raise NotImplementedError("load_csv_data function not found")


# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Map strategy short names (used in args) to classes
STRATEGY_CLASS_MAP = {
    "LongShort": LongShortStrategy,
    "MACross": MovingAverageCrossoverStrategy,
    "RSIReversion": RsiMeanReversionStrategy,
    "BBReversion": BollingerBandReversionStrategy,
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


# --- Parameter Loading Function (Similar to optimize.py but for specific values) ---
def load_best_params_from_config(
    config_path: str, symbol: str, strategy_class_name: str
) -> Optional[Dict[str, Any]]:
    """Loads the best parameters for a given symbol and strategy from the optimization config file."""
    config_file = Path(config_path)
    if not config_file.is_file():
        logger.error(f"Parameter config file not found: {config_path}")
        return None

    try:
        with open(config_file, "r") as f:
            all_configs = yaml.safe_load(f)
            if not isinstance(all_configs, dict):
                logger.error(
                    f"Invalid format in {config_path}. Expected top-level dictionary."
                )
                return None

            symbol_configs = all_configs.get(symbol)
            if not isinstance(symbol_configs, dict):
                logger.error(
                    f"Symbol {symbol} not found or invalid format in {config_path}."
                )
                return None

            strategy_params = symbol_configs.get(strategy_class_name)
            if not isinstance(strategy_params, dict):
                logger.error(
                    f"Strategy {strategy_class_name} not found for symbol {symbol} in {config_path}."
                )
                return None
    except FileNotFoundError:
        logger.error(f"Parameter config file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"Error reading parameter config {config_path}: {e}", exc_info=True
        )
        return None

    # --- Adjust for LongShortStrategy's tuple parameters ---
    if strategy_class_name == "LongShortStrategy":
        # Check if the split keys exist from the optimization save
        if (
            "return_thresh_low" in strategy_params
            and "return_thresh_high" in strategy_params
        ):
            strategy_params["return_thresh"] = (
                strategy_params.pop("return_thresh_low"),
                strategy_params.pop("return_thresh_high"),
            )
        elif "return_thresh" not in strategy_params:
            logger.error(
                f"Missing return threshold parameters for LongShortStrategy in {config_path}"
            )
            return None  # Or raise error

        if (
            "volume_thresh_low" in strategy_params
            and "volume_thresh_high" in strategy_params
        ):
            strategy_params["volume_thresh"] = (
                strategy_params.pop("volume_thresh_low"),
                strategy_params.pop("volume_thresh_high"),
            )
        elif "volume_thresh" not in strategy_params:
            logger.error(
                f"Missing volume threshold parameters for LongShortStrategy in {config_path}"
            )
            return None  # Or raise error

    # Ensure mypy knows the final type after modifications
    return cast(Optional[Dict[str, Any]], strategy_params)


# --- Function to create results directory ---
def ensure_results_dir(base_path: str) -> Path:
    results_path = Path(base_path)
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path


# --- Main Forward Testing Function ---
def run_forward_test(
    strategy_short_name: str,
    param_config_path: str,  # Path to the BEST params YAML
    symbol: str,
    data: pd.DataFrame,  # Data ALREADY potentially sliced by main()
    trade_units: float,
    commission_bps: float,
    initial_balance: float,
    # Global Filter Defaults (from command line)
    global_apply_atr_filter: bool = False,
    global_atr_filter_period: int = 14,
    global_atr_filter_multiplier: float = 1.5,
    global_atr_filter_sma_period: int = 100,
    global_apply_seasonality_filter: bool = False,
    global_allowed_trading_hours_utc: Optional[str] = None,
    global_apply_seasonality_to_symbols: Optional[str] = None,
    # Other args for logging/reporting
    fwd_test_start: Optional[str] = None,
    fwd_test_end: Optional[str] = None,
):
    """Runs a backtest on the forward test data slice using optimized parameters
    loaded from the specified configuration file.
    Now uses BacktestRunConfig.
    """
    logger.info(f"--- Running Forward Test for {strategy_short_name} on {symbol} --- ")
    logger.info(f"Loading optimized parameters from: {param_config_path}")

    # 1. Get Strategy Class Name (Needed for loading params)
    if strategy_short_name not in STRATEGY_CLASS_MAP:
        logger.error(f"Unknown strategy: {strategy_short_name}")
        return None
    strategy_class = STRATEGY_CLASS_MAP[strategy_short_name]
    strategy_class_name = strategy_class.__name__

    # 2. Load ALL Best Parameters (Strategy + SL/TP/TSL + Filters)
    best_params_all = load_best_params_from_config(
        param_config_path, symbol, strategy_class_name
    )
    if not best_params_all:
        logger.error(
            f"No best params found for {strategy_class_name} / {symbol} in {param_config_path}. Cannot run forward test."
        )
        return None
    logger.info(f"Loaded best parameters: {best_params_all}")

    # 3. Construct BacktestRunConfig
    try:
        # Extract core strategy params
        core_strategy_params = {
            k: v
            for k, v in best_params_all.items()
            if k
            not in [
                # Exclude SL/TP/TSL and known filter keys that are handled separately
                "stop_loss_pct",
                "take_profit_pct",
                "trailing_stop_loss_pct",
                "apply_atr_filter",
                "atr_filter_period",
                "atr_filter_threshold",  # Use the key from YAML
                # "atr_filter_multiplier", # Don't exclude this if it exists
                "atr_filter_sma_period",
                "apply_seasonality",  # Use the key from YAML
                "seasonality_start_hour",
                "seasonality_end_hour",
                "allowed_trading_hours_utc",  # Exclude if building separately
                "apply_seasonality_to_symbols",
            ]
        }

        # Clean None strings just in case they exist in core params
        for key, value in core_strategy_params.items():
            if isinstance(value, str) and value.lower() == "none":
                core_strategy_params[key] = None

        # --- Build Config Dictionary ---
        config_dict = {
            # Core settings from function args
            "symbol": symbol,
            "initial_balance": initial_balance,
            "commission_bps": commission_bps,
            "units": trade_units,
            # Strategy info
            "strategy_short_name": strategy_short_name,
            "strategy_params": core_strategy_params,
            # Risk params directly from loaded best params
            "stop_loss_pct": best_params_all.get("stop_loss_pct"),
            "take_profit_pct": best_params_all.get("take_profit_pct"),
            "trailing_stop_loss_pct": best_params_all.get("trailing_stop_loss_pct"),
            # Filter flags and params - prioritize from best_params file
            # If flags/params are missing from best_params, they default to False/None/model defaults
            "apply_atr_filter": best_params_all.get("apply_atr_filter", False),
            "atr_filter_period": best_params_all.get("atr_filter_period", 14),
            # Use the threshold key from YAML for multiplier
            "atr_filter_multiplier": best_params_all.get("atr_filter_threshold", 1.5),
            "atr_filter_sma_period": best_params_all.get("atr_filter_sma_period", 100),
            # Use apply_seasonality key from YAML for flag
            "apply_seasonality_filter": best_params_all.get("apply_seasonality", False),
            # Construct hours string OR take directly if saved
            "allowed_trading_hours_utc": best_params_all.get(
                "allowed_trading_hours_utc"
            ),
            "apply_seasonality_to_symbols": best_params_all.get(
                "apply_seasonality_to_symbols"
            ),
        }

        # Build hours string if start/end are present and main key isn't
        if config_dict["allowed_trading_hours_utc"] is None:
            start_hour_grid = best_params_all.get("seasonality_start_hour")
            end_hour_grid = best_params_all.get("seasonality_end_hour")
            if start_hour_grid is not None and end_hour_grid is not None:
                config_dict["allowed_trading_hours_utc"] = (
                    f"{start_hour_grid}-{end_hour_grid}"
                )

        # Validate and create the config object
        backtest_config = BacktestRunConfig(**config_dict)
        logger.info("Constructed BacktestRunConfig for forward test.")
        logger.debug(f"Forward Test Config: {backtest_config.model_dump()}")

    except ValidationError as e:
        logger.error(f"Failed to validate BacktestRunConfig for forward test: {e}")
        return None
    except Exception as e:
        logger.error(f"Error constructing BacktestRunConfig: {e}", exc_info=True)
        return None
    # --- End Construct Config --- #

    # 4. Run Backtest using the Config Object
    logger.info(
        f"Running backtest on forward test data ({fwd_test_start} to {fwd_test_end})..."
    )
    try:
        backtest_results = run_backtest(
            data=data,  # Pass the pre-sliced data
            config=backtest_config,  # Pass the config object
        )
    except Exception as e:
        logger.error(f"Exception during run_backtest call: {e}", exc_info=True)
        return None

    if not backtest_results:
        logger.error("Forward test backtest run failed or returned no results.")
        return None

    logger.info("Forward test backtest completed.")
    return backtest_results


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
        help="Short name of the strategy previously optimized.",
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
        default="config/best_params.yaml",
        help="Path to the YAML file containing the best strategy parameters.",
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
    # Pass the GLOBAL filter defaults from args to run_forward_test
    results = run_forward_test(
        strategy_short_name=args.strategy,
        param_config_path=args.param_config,  # Path to best_params.yaml
        symbol=args.symbol,
        data=forward_data_slice,  # Pass the prepared slice
        trade_units=args.units,
        commission_bps=args.commission,
        initial_balance=args.balance,
        # Pass GLOBAL defaults from CLI args
        global_apply_atr_filter=args.apply_atr_filter,
        global_atr_filter_period=args.atr_filter_period,
        global_atr_filter_multiplier=args.atr_filter_multiplier,
        global_atr_filter_sma_period=args.atr_filter_sma_period,
        global_apply_seasonality_filter=args.apply_seasonality_filter,
        global_allowed_trading_hours_utc=args.allowed_trading_hours_utc,
        global_apply_seasonality_to_symbols=args.apply_seasonality_to_symbols,
        # SL/TP/TSL will be loaded from file inside run_forward_test
        fwd_test_start=args.fwd_start,  # Pass for logging/reporting
        fwd_test_end=args.fwd_end,  # Pass for logging/reporting
    )

    # --- Process Results ---
    if results is None:
        logger.error("Forward test execution failed or produced no results.")
        exit(1)

    perf_summary = results.get("performance_summary")
    trades_df = results.get("trades")
    equity_curve = results.get("equity_curve")

    if perf_summary is None or trades_df is None or equity_curve is None:
        logger.error("Forward test returned incomplete results dictionary.")
        exit(1)

    logger.info("\n--- Forward Test Performance Summary ---")
    logger.info(f"\n{perf_summary.to_string()}")

    # --- Save Results ---
    if args.plotting or trades_df is not None:
        results_path = ensure_results_dir(args.results_dir)
        trades_path = results_path / "trades"
        trades_path.mkdir(exist_ok=True)

        # Generate a unique filename base
        fwd_end_str = args.fwd_end.replace("-", "") if args.fwd_end else "latest"
        fwd_start_str = args.fwd_start.replace("-", "")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{args.strategy}_{args.symbol}_fwd_{fwd_start_str}_{fwd_end_str}_{timestamp}"

        # Add filter info to filename
        filter_suffix = ""
        if args.apply_atr_filter:
            filter_suffix += f"_atr{args.atr_filter_period}x{args.atr_filter_multiplier:.1f}sma{args.atr_filter_sma_period}"
        if args.apply_seasonality_filter:
            filter_suffix += f"_season{args.allowed_trading_hours_utc.replace('-','')}"
            if args.apply_seasonality_to_symbols:
                syms = args.apply_seasonality_to_symbols.split(",")
                filter_suffix += f"_{''.join(s[:3] for s in syms)}"

        # Sanitize suffix
        from .optimize import sanitize_filename  # Borrow sanitize function

        filter_suffix = sanitize_filename(filter_suffix)
        base_filename += filter_suffix

        # Save Trades CSV
        if trades_df is not None and not trades_df.empty:
            trades_csv_path = trades_path / f"{base_filename}_trades.csv"
            trades_df.to_csv(trades_csv_path, index=False)
            logger.info(f"Forward test trades saved to: {trades_csv_path}")
        else:
            logger.info("No trades executed during the forward test.")

        # Generate Plots and Report (if plotting enabled)
        if args.plotting and equity_curve is not None:
            plot_path = results_path / "plots"
            plot_path.mkdir(exist_ok=True)
            report_path = results_path / "reports"
            report_path.mkdir(exist_ok=True)

            plot_filename = plot_path / f"{base_filename}_equity.png"
            plot_equity_curve(
                equity_curve=equity_curve,
                plot_filename=plot_filename,
                title_prefix=f"{args.strategy} {args.symbol} Forward Test",
            )

            # Generate HTML Report
            report_filename = report_path / f"{base_filename}_forward_test_report.html"

            # --- Prepare parameters for the report (using best_params) ---
            strategy_class = STRATEGY_CLASS_MAP.get(args.strategy)
            strategy_class_name = (
                strategy_class.__name__ if strategy_class else "UnknownStrategy"
            )
            best_params = (
                load_best_params_from_config(
                    config_path=args.param_config,
                    symbol=args.symbol,
                    strategy_class_name=strategy_class_name,
                )
                or {}
            )

            # Get core strategy params (exclude risk/filter keys)
            strategy_params_for_report = {
                k: v
                for k, v in best_params.items()
                if k
                not in [
                    "stop_loss_pct",
                    "take_profit_pct",
                    "trailing_stop_loss_pct",
                    "apply_atr_filter",
                    "atr_filter_period",
                    "atr_filter_threshold",
                    "atr_filter_multiplier",
                    "atr_filter_sma_period",
                    "apply_seasonality",
                    "seasonality_start_hour",
                    "seasonality_end_hour",
                    "allowed_trading_hours_utc",
                    "apply_seasonality_to_symbols",
                ]
            }
            # Clean None strings
            for key, value in strategy_params_for_report.items():
                if isinstance(value, str) and value.lower() == "none":
                    strategy_params_for_report[key] = None

            # Get test parameters (use best_params for filters)
            test_params_for_report = {
                "Symbol": args.symbol,
                # "Interval": args.interval or "N/A", # Interval not available in args?
                "Forward Start": args.fwd_start,
                "Forward End": args.fwd_end or "End of Data",
                "Initial Balance": args.balance,
                "Commission (bps)": args.commission,
                "Units": args.units,
                # --- Derive Filter status *from best_params* --- #
                "ATR Filter Enabled": best_params.get("apply_atr_filter", False),
                "Seasonality Filter Enabled": best_params.get(
                    "apply_seasonality", False
                ),
            }

            # Add specific filter params ONLY if the filter was enabled according to best_params
            if test_params_for_report["ATR Filter Enabled"]:
                test_params_for_report["ATR Filter Period"] = best_params.get(
                    "atr_filter_period",
                    "N/A (Missing in best_params)",  # Report missing
                )
                test_params_for_report["ATR Filter Multiplier"] = best_params.get(
                    "atr_filter_threshold",
                    "N/A (Missing in best_params)",  # Use threshold key
                )
                test_params_for_report["ATR Filter SMA Period"] = best_params.get(
                    "atr_filter_sma_period", "N/A (Missing in best_params)"
                )
            if test_params_for_report["Seasonality Filter Enabled"]:
                # Try constructing from start/end hours first
                start_h = best_params.get("seasonality_start_hour")
                end_h = best_params.get("seasonality_end_hour")
                hours_str = best_params.get(
                    "allowed_trading_hours_utc"
                )  # Check if combined key exists
                if hours_str is None and start_h is not None and end_h is not None:
                    hours_str = f"{start_h}-{end_h}"

                test_params_for_report["Allowed Trading Hours"] = (
                    hours_str
                    if hours_str is not None
                    else "N/A (Missing in best_params)"
                )
                test_params_for_report["Seasonality Applied Symbols"] = best_params.get(
                    "apply_seasonality_to_symbols", "N/A (Missing in best_params)"
                )
            # --- End Deriving Filter status --- #

            # Extract relevant metrics for the report (assuming run_backtest returns these)
            metrics_for_report = {
                "cumulative_profit": results.get("cumulative_profit", 0.0),
                "final_balance": results.get("final_balance", args.balance),
                "total_trades": results.get("total_trades", 0),
                "win_rate": results.get("win_rate", 0.0),
                "sharpe_ratio": results.get("sharpe_ratio", 0.0),
                "max_drawdown": results.get("max_drawdown", 0.0),
                "profit_factor": results.get("profit_factor", 0.0),
            }

            generate_html_report(
                metrics=metrics_for_report,
                plot_path=plot_filename,
                report_filename=report_filename,
                report_title=f"{args.strategy} {args.symbol} Forward Test Report",
                test_params=test_params_for_report,
                strategy_params=strategy_params_for_report,  # Use the cleaned core params
            )
            logger.info(f"Forward test report saved to: {report_filename}")
        elif not args.plotting:
            logger.info("Plotting disabled by user.")

    logger.info("--- Forward test script finished. ---")
