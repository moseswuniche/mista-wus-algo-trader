import argparse
import pandas as pd
from pathlib import Path
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np  # Add numpy for std dev calculation
import matplotlib.pyplot as plt  # Add matplotlib
import string  # For filename sanitization
import os

# Import shared reporting utils
from .reporting_utils import plot_equity_curve, generate_html_report

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_trade_filename(filepath: Path) -> Optional[Dict[str, str]]:
    """Parses strategy, symbol, and params_str from a trade filename.

    Expected format: SYMBOL_STRATEGY_PARAMSTRING_trades.csv
    Example: BTCUSDT_MACross_ma_short_5_ma_long_20_trades.csv
    """
    name = (
        filepath.stem
    )  # Get filename without extension (e.g., BTCUSDT_MACross_ma_short_5_ma_long_20_trades)
    # Remove the trailing '_trades'
    if name.endswith("_trades"):
        name = name[: -len("_trades")]
    else:
        logger.warning(
            f"Filename {filepath.name} does not end with '_trades'. Skipping."
        )
        return None

    parts = name.split("_")
    if len(parts) < 3:
        logger.warning(
            f"Cannot parse filename {filepath.name}. Expected at least SYMBOL_STRATEGY_PARAMS format. Found {len(parts)} parts."
        )
        return None

    # Assume symbol is the first part (might contain numbers, e.g., 1000SHIBUSDT)
    symbol = parts[0]

    # Find the strategy name - assumes it's one of the known keys or similar
    # This is heuristic - might need adjustment if strategy names have underscores
    # Let's find the first part that looks like a parameter key (contains a number or common key names)
    param_start_index = -1
    potential_strategy_parts = []
    known_param_hints = [
        "period",
        "std",
        "thresh",
        "window",
        "ma",
        "rsi",
        "bb",
        "sl",
        "tp",
        "tsl",
        "loss",
        "profit",
        "limit",
        "stop",
    ]

    # Iterate starting from the second part
    for i in range(1, len(parts)):
        part = parts[i]
        # Check if part looks like a param key=value start OR is purely numeric
        # Need double backslash for regex escape within f-string - Correction: No f-string here, r'' handles it.
        is_param_key = any(
            hint in part.lower() for hint in known_param_hints
        ) or re.match(
            r"p\\d+", part
        )  # Check for common param names or pX from skopt (Use r'' raw string for regex)
        is_numeric_value = (
            parts[i + 1].replace(".", "", 1).isdigit()
            if (i + 1 < len(parts))
            else False
        )

        if is_param_key and is_numeric_value:
            param_start_index = i
            break
        else:
            potential_strategy_parts.append(part)

    if not potential_strategy_parts or param_start_index == -1:
        logger.warning(
            f"Could not reliably distinguish strategy from parameters in {filepath.name}. Skipping."
        )
        return None

    strategy = "_".join(potential_strategy_parts)
    params_str = "_".join(parts[param_start_index:])

    return {
        "symbol": symbol,
        "strategy": strategy,
        "params_str": params_str,
        "filepath": str(filepath),
    }


def parse_details_filename(filepath: Path) -> Optional[Dict[str, str]]:
    """Parses strategy and symbol from an optimize_details filename."""
    # Example: MACross_SUIUSDT_optimize_details_20250428_152614.csv
    name = filepath.stem
    parts = name.split("_")
    # Check specific indices for 'optimize' and 'details' assuming STRATEGY_SYMBOL_optimize_details_DATE_TIME structure
    if len(parts) < 6 or parts[2] != "optimize" or parts[3] != "details":
        logger.warning(
            f"Cannot parse details filename {filepath.name}. Expected STRATEGY_SYMBOL_optimize_details_DATE_TIME format (found {len(parts)} parts)."
        )
        return None

    strategy = parts[0]
    symbol = parts[1]
    return {"strategy": strategy, "symbol": symbol, "filepath": str(filepath)}


def find_files(
    results_dir: Path,
    strategy_filter: Optional[str],
    symbol_filter: Optional[str],
    analyze_details: bool,
) -> List[Dict[str, str]]:
    """Finds trade CSVs or optimize detail CSVs, parses metadata, and filters."""
    files_metadata = []

    if analyze_details:
        search_pattern = "*_optimize_details_*.csv"
        search_root = results_dir  # Details files are expected in the root results_dir
        parser_func = parse_details_filename
        file_type = "optimize details"
        logger.info(
            f"Searching for '{search_pattern}' in '{search_root}' (Details Mode)..."
        )
    else:
        search_pattern = "*_trades.csv"
        # Search recursively within the trades subdirectory if it exists
        trades_subdir = results_dir / "trades"
        search_root = (
            trades_subdir if trades_subdir.is_dir() else results_dir
        )  # Fallback to root dir
        parser_func = parse_trade_filename
        file_type = "trade logs"
        logger.info(
            f"Searching for '{search_pattern}' in '{search_root}' recursively (Trade Log Mode)..."
        )

    for filepath in search_root.rglob(search_pattern):
        metadata = parser_func(filepath)
        if metadata:
            # Apply filters
            if strategy_filter and metadata["strategy"] != strategy_filter:
                continue
            if symbol_filter and metadata["symbol"] != symbol_filter:
                continue
            files_metadata.append(metadata)

    logger.info(f"Found {len(files_metadata)} matching {file_type} files.")
    return files_metadata


def load_and_process_data(
    files_metadata: List[Dict[str, str]], analyze_details: bool
) -> Optional[pd.DataFrame]:
    """Loads and combines data, either trade logs or optimization details."""
    all_data_list = []

    if not files_metadata:
        logger.warning("No files found or parsed successfully. Cannot load data.")
        return None

    if analyze_details:
        # --- Load Optimize Details Files ---
        for metadata in files_metadata:
            filepath = Path(metadata["filepath"])
            try:
                df = pd.read_csv(filepath)
                # Basic validation - parenthesis around the condition for clarity
                if not (
                    any(col.startswith("params.") for col in df.columns)
                    or any(col.startswith("result_") for col in df.columns)
                ):
                    logger.warning(
                        f"Skipping details file {filepath.name}: Missing expected 'params.*' or 'result_.*' columns."
                    )
                    continue

                # Add strategy/symbol from filename metadata
                df["strategy"] = metadata["strategy"]
                df["symbol"] = metadata["symbol"]
                df["source_file"] = filepath.name
                all_data_list.append(df)
            except Exception as e:
                logger.error(
                    f"Error loading details file {filepath.name}: {e}", exc_info=True
                )

        if not all_data_list:
            logger.error("No optimize details data could be loaded.")
            return None

        combined_df = pd.concat(all_data_list, ignore_index=True)
        logger.info(
            f"Combined details data from {len(all_data_list)} files into a DataFrame with {len(combined_df)} rows (parameter sets)."
        )
        return combined_df  # Return the combined details dataframe directly

    else:
        # --- Load Trade Log Files (Original Logic) ---
        for metadata in files_metadata:
            filepath = Path(metadata["filepath"])
            try:
                df = pd.read_csv(filepath)
                df["symbol"] = metadata["symbol"]
                df["strategy"] = metadata["strategy"]
                df["params_str"] = metadata["params_str"]
                df["source_file"] = filepath.name
                required_cols = ["Entry Time", "Exit Time", "PnL"]
                if not all(col in df.columns for col in required_cols):
                    logger.warning(
                        f"Skipping {filepath.name}: Missing required columns ({required_cols}). Found: {list(df.columns)}"
                    )
                    continue
                all_data_list.append(df)
            except pd.errors.EmptyDataError:
                logger.warning(f"Skipping empty trade file: {filepath.name}")
            except Exception as e:
                logger.error(
                    f"Error loading trade file {filepath.name}: {e}", exc_info=True
                )

        if not all_data_list:
            logger.error("No trade data could be loaded from the found files.")
            return None

        combined_df = pd.concat(all_data_list, ignore_index=True)
        logger.info(
            f"Combined trade data from {len(all_data_list)} files into a DataFrame with {len(combined_df)} trades."
        )

        # Convert time columns (only for trade logs)
        try:
            combined_df["Entry Time"] = pd.to_datetime(combined_df["Entry Time"])
            combined_df["Exit Time"] = pd.to_datetime(combined_df["Exit Time"])
            combined_df["Duration"] = (
                combined_df["Exit Time"] - combined_df["Entry Time"]
            )
        except Exception as e:
            logger.error(f"Error converting time columns to datetime: {e}")
            combined_df["Duration"] = pd.NaT

        return combined_df  # Return the combined trades dataframe


def calculate_metrics(trades_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates performance metrics for a DataFrame of trades."""
    metrics: Dict[str, Any] = {}  # Explicitly type the metrics dictionary
    total_trades = len(trades_df)
    metrics["total_trades"] = total_trades

    # Default values for metrics if no trades
    # Use 0.0 for potentially float values
    default_metrics = {
        "total_pnl": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_pnl": 0.0,
        "avg_duration": pd.Timedelta(0),
        "avg_win_pnl": 0.0,
        "avg_loss_pnl": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
    }

    if total_trades == 0:
        metrics.update(default_metrics)
        return metrics

    # --- Basic PnL & Win Rate ---
    metrics["total_pnl"] = trades_df["PnL"].sum()
    metrics["avg_pnl"] = trades_df["PnL"].mean()

    winning_trades_df = trades_df[trades_df["PnL"] > 0]
    losing_trades_df = trades_df[trades_df["PnL"] < 0]
    metrics["win_rate"] = (
        (len(winning_trades_df) / total_trades) * 100 if total_trades > 0 else 0.0
    )  # Ensure float

    # --- Average Win/Loss ---
    metrics["avg_win_pnl"] = (
        winning_trades_df["PnL"].mean() if not winning_trades_df.empty else 0.0
    )  # Ensure float
    metrics["avg_loss_pnl"] = (
        losing_trades_df["PnL"].mean() if not losing_trades_df.empty else 0.0
    )  # Ensure float

    # --- Profit Factor ---
    total_profit = winning_trades_df["PnL"].sum()
    total_loss = abs(losing_trades_df["PnL"].sum())
    metrics["profit_factor"] = (
        total_profit / total_loss if total_loss > 0 else float("inf")
    )

    # --- Duration ---
    if "Duration" in trades_df.columns and trades_df["Duration"].notna().any():
        metrics["avg_duration"] = trades_df["Duration"].mean()
    else:
        metrics["avg_duration"] = pd.NaT

    # --- Max Drawdown (based on cumulative PnL sequence) ---
    # Note: Assumes trades_df is ordered chronologically or sequentially as executed
    cumulative_pnl = trades_df["PnL"].cumsum()
    peak = cumulative_pnl.cummax()
    drawdown = peak - cumulative_pnl
    metrics["max_drawdown"] = (
        drawdown.max() if not drawdown.empty else 0.0
    )  # Ensure float

    # --- Simplified Sharpe & Sortino Ratios (per trade, risk-free=0) ---
    pnl_std_dev = trades_df["PnL"].std()
    metrics["sharpe_ratio"] = (
        metrics["avg_pnl"] / pnl_std_dev if pnl_std_dev and pnl_std_dev > 0 else 0.0
    )  # Check pnl_std_dev exists and > 0, default float

    downside_std_dev = (
        losing_trades_df["PnL"].std() if not losing_trades_df.empty else 0.0
    )  # Default float
    # Ensure avg_pnl exists before comparison
    avg_pnl_val = metrics.get("avg_pnl", 0.0)
    metrics["sortino_ratio"] = (
        avg_pnl_val / downside_std_dev
        if downside_std_dev > 0
        else float("inf") if avg_pnl_val > 0 else 0.0
    )  # Default float

    # Replace NaNs that might occur (e.g., std dev of 0 or 1 trade)
    for key in [
        "avg_win_pnl",
        "avg_loss_pnl",
        "max_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
    ]:
        # Check metrics[key] exists before pd.isna
        if key in metrics and pd.isna(metrics[key]):
            # Use 0.0 for float metrics
            metrics[key] = (
                0.0
                if key != "sortino_ratio"
                else (float("inf") if metrics.get("avg_pnl", 0.0) > 0 else 0.0)
            )

    return metrics


def format_metrics(metrics: Dict[str, Any]) -> Dict[str, str]:
    """Formats calculated metrics for display."""
    sharpe = metrics.get("sharpe_ratio", 0)
    sortino = metrics.get("sortino_ratio", 0)

    return {
        "Total Trades": f"{metrics.get('total_trades', 0)}",
        "Total PnL": f"{metrics.get('total_pnl', 0):.4f}",
        "Avg PnL": f"{metrics.get('avg_pnl', 0):.4f}",
        "Avg Win PnL": f"{metrics.get('avg_win_pnl', 0):.4f}",
        "Avg Loss PnL": f"{metrics.get('avg_loss_pnl', 0):.4f}",
        "Win Rate (%)": f"{metrics.get('win_rate', 0):.2f}",
        "Profit Factor": (
            f"{metrics.get('profit_factor', 0):.2f}"
            if metrics.get("profit_factor") != float("inf")
            else "Inf"
        ),
        "Max Drawdown": f"{metrics.get('max_drawdown', 0):.4f}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}" if sortino != float("inf") else "Inf",
        "Avg Duration": str(metrics.get("avg_duration", pd.NaT)),
    }


# <<< Helper Function for Filename Sanitization (copied from optimize.py) >>>
def sanitize_filename(filename: str) -> str:
    """Removes or replaces characters invalid for filenames."""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    sanitized = "".join(c if c in valid_chars else "_" for c in filename)
    return sanitized


# <<< Helper Function to Create Parameter String (copied from optimize.py) >>>
def create_param_string(params: Dict[str, Any]) -> str:  # type: ignore[return]
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


# <<< Function to plot results for a SPECIFIC GROUP >>>
def plot_group_results(
    group_df: pd.DataFrame, group_name_tuple: Tuple[str, str, str], plot_dir: Path
) -> Optional[Path]:
    """Generates and saves equity curve plot for a single group."""
    strategy, symbol, params_str = group_name_tuple

    # Sanitize param string for filename
    sanitized_params = sanitize_filename(params_str)
    # Create a unique filename
    base_plot_filename = f"{strategy}_{symbol}_{sanitized_params}_equity.png"
    # Create subdirectories
    group_plot_dir = plot_dir / strategy / symbol
    plot_filepath = group_plot_dir / base_plot_filename

    if group_df.empty:
        logger.warning(f"Group data empty for {group_name_tuple}. Skipping plot.")
        return None

    try:
        # Ensure data is sorted by time for a meaningful cumulative plot
        if "Entry Time" in group_df.columns:
            plot_data = group_df.sort_values("Entry Time").copy()
            plot_data["cumulative_pnl"] = plot_data["PnL"].cumsum()
            # Check if cumulative_pnl series is valid before plotting
            if plot_data["cumulative_pnl"].notna().any():
                equity_curve = plot_data["cumulative_pnl"]  # Use this for plotting
                title_prefix = f"{strategy} {symbol} ({params_str[:30]}...)"
                # Call the shared plotting function
                if plot_equity_curve(
                    equity_curve, plot_filepath, title_prefix=title_prefix
                ):
                    return plot_filepath  # Return path if plot saved successfully
                else:
                    return None
            else:
                logger.warning(
                    f"Cumulative PnL is all NaN for group {group_name_tuple}. Skipping plot."
                )
                return None
        else:
            logger.warning(
                f"'Entry Time' column missing for group {group_name_tuple}. Cannot generate equity plot."
            )
            return None
    except Exception as e:
        logger.error(
            f"Error plotting group results for {group_name_tuple}: {e}", exc_info=True
        )
        return None


# <<< Function to plot OVERALL results >>>
def plot_overall_results(
    combined_df: Optional[pd.DataFrame],
    results_df: pd.DataFrame,
    top_n_results: Optional[pd.DataFrame],
    args: argparse.Namespace,
) -> Optional[Path]:
    """Generates and saves plots based on analysis results. Returns path to Top N plot if generated."""
    if not args.plotting:
        logger.info("Plotting is disabled by command-line argument.")
        return None

    plot_dir = Path(args.plot_dir)  # Use plot_dir from args
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving overall plots to directory: {plot_dir}")

    # Define a common filename prefix based on filters
    filename_prefix = (
        f"overall_analysis_{args.strategy or 'all'}_{args.symbol or 'all'}"
    )

    top_n_plot_path: Optional[Path] = None  # Variable to store path of the top N plot

    # --- Plots requiring combined_df (raw trade data) ---
    if combined_df is not None and not combined_df.empty:
        # 1. Overall PnL Distribution (Histogram)
        try:
            plt.figure(figsize=(10, 6))
            combined_df["PnL"].hist(bins=50)
            plt.title(
                f"Overall PnL Distribution ({args.strategy or 'All Strategies'}, {args.symbol or 'All Symbols'})"
            )
            plt.xlabel("PnL per Trade")
            plt.ylabel("Frequency")
            plt.grid(axis="y", alpha=0.75)
            plot_path = plot_dir / f"{filename_prefix}_pnl_distribution.png"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved Overall PnL distribution plot to {plot_path}")
        except Exception as e:
            logger.error(
                f"Failed to generate/save Overall PnL distribution plot: {e}",
                exc_info=True,
            )

        # 2. Overall Cumulative PnL (Line Plot) - Use shared plot_equity_curve
        try:
            if "Entry Time" in combined_df.columns:
                plot_df = combined_df.sort_values("Entry Time").copy()
                plot_df["cumulative_pnl"] = plot_df["PnL"].cumsum()
                if plot_df["cumulative_pnl"].notna().any():
                    equity_curve = plot_df["cumulative_pnl"]
                    plot_filepath = plot_dir / f"{filename_prefix}_cumulative_pnl.png"
                    plot_equity_curve(
                        equity_curve,
                        plot_filepath,
                        title_prefix=f"Overall {args.strategy or 'All'} {args.symbol or 'All'}",
                    )
                else:
                    logger.warning("Overall cumulative PnL is all NaN. Skipping plot.")
            else:
                logger.warning(
                    "'Entry Time' column missing. Cannot generate overall equity plot."
                )
        except Exception as e:
            logger.error(
                f"Failed to generate/save overall cumulative PnL plot: {e}",
                exc_info=True,
            )
    elif args.plotting:  # Log warning only if plotting was intended
        logger.warning(
            "Raw trade data not available (or empty). Skipping PnL distribution and overall equity plots."
        )

    # --- Plots requiring top_n_results (aggregated data) ---
    # 3. Top N Parameter Sets Performance (Bar Chart)
    if top_n_results is not None and not top_n_results.empty:
        try:
            plt.figure(figsize=(12, 7))
            # Create a shorter label for the bars
            top_n_results["param_label"] = top_n_results.apply(
                lambda row: f"{row['strategy']}_{row['symbol']}\n{(row['params_str'] or '')[:30]}{'...' if row['params_str'] and len(row['params_str']) > 30 else ''}",
                axis=1,
            )
            bars = plt.bar(
                top_n_results["param_label"], top_n_results["total_pnl"]
            )  # Use raw pnl for height
            plt.title(
                f"Top {len(top_n_results)} Parameter Sets by Total PnL (Sorted by Profit Factor then PnL)"
            )
            plt.xlabel("Strategy & Parameters")
            plt.ylabel("Total PnL")
            plt.xticks(rotation=60, ha="right", fontsize=8)
            plt.grid(axis="y", alpha=0.75)
            # Add PnL value labels on bars
            plt.bar_label(bars, fmt="{:,.2f}", fontsize=8)
            top_n_plot_path = (
                plot_dir / f"{filename_prefix}_top_n_pnl_comparison.png"
            )  # Assign path here
            plt.tight_layout()
            plt.savefig(top_n_plot_path)
            plt.close()
            logger.info(f"Saved Top N comparison plot to {top_n_plot_path}")
        except Exception as e:
            logger.error(
                f"Failed to generate/save Top N comparison plot: {e}", exc_info=True
            )
            top_n_plot_path = None  # Ensure path is None on error
    elif args.plotting:
        logger.warning(
            "Top N results data not available (or empty). Skipping Top N comparison plot."
        )

    return top_n_plot_path  # Return the path to the top N plot


def analyze_trades(input_df: pd.DataFrame, top_n: int = 10, args=None):
    """Performs analysis on trade logs or optimization details."""

    if input_df is None or input_df.empty:
        logger.info("No data to analyze.")
        return

    # Determine mode based on args flag
    analyze_details_mode = getattr(args, "analyze_details", False)

    if analyze_details_mode:
        logger.info("--- Analyzing Optimization Details --- ")
        # Data is already aggregated metrics per parameter set

        # Rename columns from details file to expected metric names
        rename_map = {
            "result_cumulative_profit": "total_pnl",
            "result_total_trades": "total_trades",
            "result_win_rate": "win_rate",
            "result_profit_factor": "profit_factor",
            "result_sharpe_ratio": "sharpe_ratio",
            "result_max_drawdown": "max_drawdown",
            "result_sortino_ratio": "sortino_ratio",  # Assuming Sortino is saved
            # Add mappings for avg_pnl, avg_win_pnl, avg_loss_pnl if they exist in details file
        }
        # Only rename columns that actually exist in the dataframe
        existing_rename_map = {
            k: v for k, v in rename_map.items() if k in input_df.columns
        }
        results_df = input_df.rename(columns=existing_rename_map)

        # Create 'params_str' from 'params.*' columns
        param_cols = [col for col in results_df.columns if col.startswith("params.")]

        def create_param_str(row):
            params_dict = {col.split(".", 1)[1]: row[col] for col in param_cols}
            return create_param_string(
                params_dict
            )  # Use helper from optimize.py (assuming it's available or copied here)

        # Make sure helper function `create_param_string` is accessible
        # If not defined in this file, we need to define or import it.
        # For now, assume `create_param_string` (similar to optimize.py's) is defined above.
        results_df["params_str"] = results_df.apply(create_param_str, axis=1)

        # Fill missing metrics with defaults (e.g., avg win/loss might not be in details)
        metric_defaults = {
            "avg_pnl": 0.0,
            "avg_win_pnl": 0.0,
            "avg_loss_pnl": 0.0,
            "avg_duration": pd.NaT,
        }
        for col, default_val in metric_defaults.items():
            if col not in results_df.columns:
                results_df[col] = default_val

        # Ensure required columns exist for sorting/display
        if "total_pnl" not in results_df.columns:
            results_df["total_pnl"] = 0.0
        if "profit_factor" not in results_df.columns:
            results_df["profit_factor"] = 0.0
        # Handle potential NaNs in crucial columns
        results_df["profit_factor"] = results_df["profit_factor"].fillna(0.0)
        results_df["total_pnl"] = results_df["total_pnl"].fillna(0.0)

        # Sort results
        results_df.sort_values(
            by=["profit_factor", "total_pnl"], ascending=[False, False], inplace=True
        )

        # Overall metrics aren't really applicable in this mode (each row is a result)
        logger.info("Skipping overall metrics calculation in details analysis mode.")
        combined_trades_df_for_plotting = None  # No combined raw trades available

    else:
        # --- Original Trade Log Analysis Logic ---
        logger.info("--- Analyzing Trade Logs --- ")
        combined_trades_df_for_plotting = input_df  # Use input df for overall plotting

        # --- Overall Analysis (Only for trade logs) ---
        logger.info("--- Overall Performance (All Trades) ---")
        if "Entry Time" in combined_trades_df_for_plotting.columns:
            combined_trades_df_for_plotting.sort_values("Entry Time", inplace=True)
        overall_metrics = calculate_metrics(combined_trades_df_for_plotting)
        formatted_overall = format_metrics(overall_metrics)
        for key, value in formatted_overall.items():
            logger.info(f"{key}: {value}")

        # --- Grouped Analysis (Calculate metrics per group) ---
        logger.info(f"\n--- Analyzing Performance by Parameter Set ---")
        group_cols = ["strategy", "symbol", "params_str"]
        if not all(
            col in combined_trades_df_for_plotting.columns for col in group_cols
        ):
            logger.error(
                f"Missing required columns for grouping: {group_cols}. Found: {list(combined_trades_df_for_plotting.columns)}"
            )
            return

        grouped = combined_trades_df_for_plotting.groupby(group_cols)
        results = []
        processed_groups = 0

        report_dir = Path(args.report_dir)  # Get report dir from args
        plot_dir = Path(args.plot_dir)  # Get plot dir from args

        for name, group in grouped:
            strategy, symbol, params_str = name  # Unpack group name tuple
            logger.debug(f"Processing group: {strategy} | {symbol} | {params_str}")

            if "Entry Time" in group.columns:
                group = group.sort_values("Entry Time").copy()
            else:
                group = group.copy()

            metrics = calculate_metrics(group)
            result_row = {
                "strategy": strategy,
                "symbol": symbol,
                "params_str": params_str,
                **metrics,
            }
            results.append(result_row)

            # --- Generate Per-Group Report & Plots (if plotting enabled) ---
            if args.plotting:
                # Generate group-specific plot (only possible if we have trade data)
                group_plot_path = plot_group_results(group, name, plot_dir)

                report_metrics = metrics
                report_run_params = {"strategy": strategy, "symbol": symbol}
                try:
                    param_parts = params_str.split("_")
                    report_strategy_params = dict(
                        zip(param_parts[::2], param_parts[1::2])
                    )
                except Exception:
                    report_strategy_params = {"params_str": params_str}

                sanitized_params = sanitize_filename(params_str)
                report_base_filename = (
                    f"{strategy}_{symbol}_{sanitized_params}_report.html"
                )
                group_report_dir = report_dir / strategy / symbol
                report_filepath = group_report_dir / report_base_filename
                report_title = (
                    f"Analysis Report: {strategy} {symbol} ({params_str[:30]}...)"
                )

                generate_html_report(
                    metrics=report_metrics,
                    plot_path=group_plot_path,
                    report_filename=report_filepath,
                    report_title=report_title,
                    test_params=report_run_params,
                    strategy_params=report_strategy_params,
                )
            processed_groups += 1

        logger.info(f"Processed {processed_groups} parameter set groups.")

        if not results:
            logger.info("No results generated after grouping.")
            return

        results_df = pd.DataFrame(results)
        results_df.sort_values(
            by=["profit_factor", "total_pnl"], ascending=[False, False], inplace=True
        )

    # --- COMMON SECTION: Display Top N, Save Summary, Plot Overall ---

    top_n_plot_file: Optional[Path] = None  # Initialize plot file path

    # Display Top N Results (Applies to both modes)
    top_results = results_df.head(top_n)
    if top_results.empty:
        logger.info("No results to display after sorting.")
    else:
        # Format for display
        display_cols = [
            "strategy",
            "symbol",
            "params_str",
            "total_trades",
            "total_pnl",
            "win_rate",
            "profit_factor",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "avg_pnl",
            "avg_win_pnl",
            "avg_loss_pnl",
        ]
        # Filter display_cols based on columns actually present in results_df
        available_display_cols = [
            col for col in display_cols if col in results_df.columns
        ]
        top_results_display = top_results[available_display_cols].copy()

        # Format numerical columns
        format_float_4dp = lambda x: (
            f"{x:.4f}" if isinstance(x, (int, float)) else str(x)
        )
        format_float_2dp = lambda x: (
            f"{x:.2f}" if isinstance(x, (int, float)) else str(x)
        )
        format_percent = lambda x: (
            f"{x:.2f}%" if isinstance(x, (int, float)) else str(x)
        )
        format_inf = lambda x: (
            f"{x:.2f}"
            if isinstance(x, (int, float)) and x != float("inf")
            else ("Inf" if x == float("inf") else str(x))
        )

        if "total_pnl" in top_results_display:
            top_results_display["total_pnl"] = top_results_display["total_pnl"].map(
                format_float_4dp
            )
        if "avg_pnl" in top_results_display:
            top_results_display["avg_pnl"] = top_results_display["avg_pnl"].map(
                format_float_4dp
            )
        if "avg_win_pnl" in top_results_display:
            top_results_display["avg_win_pnl"] = top_results_display["avg_win_pnl"].map(
                format_float_4dp
            )
        if "avg_loss_pnl" in top_results_display:
            top_results_display["avg_loss_pnl"] = top_results_display[
                "avg_loss_pnl"
            ].map(format_float_4dp)
        if "max_drawdown" in top_results_display:
            top_results_display["max_drawdown"] = top_results_display[
                "max_drawdown"
            ].map(format_float_4dp)
        if "win_rate" in top_results_display:
            top_results_display["win_rate"] = top_results_display["win_rate"].map(
                format_percent
            )
        if "profit_factor" in top_results_display:
            top_results_display["profit_factor"] = top_results_display[
                "profit_factor"
            ].map(format_inf)
        if "sharpe_ratio" in top_results_display:
            top_results_display["sharpe_ratio"] = top_results_display[
                "sharpe_ratio"
            ].map(format_float_2dp)
        if "sortino_ratio" in top_results_display:
            top_results_display["sortino_ratio"] = top_results_display[
                "sortino_ratio"
            ].map(format_inf)

        logger.info(
            f"\n--- Top {len(top_results_display)} Parameter Sets (Sorted by Profit Factor, then Total PnL) ---"
        )
        logger.info(f"\n{top_results_display.to_string(index=False)}\n")

        # --- Generate Plot (Top N for details mode, All for trades mode) ---
        if args.plotting:
            if analyze_details_mode:
                # Only generate the Top N plot in details mode
                top_n_plot_file = plot_overall_results(
                    None, results_df, top_results, args
                )
            else:
                # Generate all relevant plots in trade log mode
                top_n_plot_file = plot_overall_results(
                    combined_trades_df_for_plotting, results_df, top_results, args
                )

        # --- Generate HTML report for Details Analysis (if plotting enabled) ---
        if args.plotting and analyze_details_mode:
            try:
                report_dir = Path(args.report_dir)
                report_dir.mkdir(parents=True, exist_ok=True)
                # Use filters in filename if provided
                filename_suffix = f"_{args.strategy or 'all'}_{args.symbol or 'all'}"
                report_filepath = (
                    report_dir
                    / f"optimization_details_summary{filename_suffix}_report.html"
                )

                # --- Calculate Relative Plot Path ---
                plot_img_tag = ""
                if top_n_plot_file:
                    try:
                        # Calculate relative path from report directory to plot file
                        rel_plot_path = Path(
                            os.path.relpath(top_n_plot_file, report_dir)
                        )
                        plot_img_tag = f'<img src="{rel_plot_path}" alt="Top N Performance Plot" style="max-width: 100%; height: auto;">'
                    except ValueError as e:
                        logger.warning(
                            f"Could not determine relative path for plot {top_n_plot_file} from {report_dir}. Using absolute path. Error: {e}"
                        )
                        plot_img_tag = f'<img src="{top_n_plot_file.resolve()}" alt="Top N Performance Plot" style="max-width: 100%; height: auto;">'  # Fallback to absolute
                    except Exception as e:
                        logger.error(
                            f"Error creating relative plot path: {e}", exc_info=True
                        )

                # Basic HTML structure
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Optimization Details Summary ({args.strategy or 'All Strategies'}, {args.symbol or 'All Symbols'})</title>
                    <style>
                        body {{ font-family: sans-serif; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    </style>
                </head>
                <body>
                    <h1>Optimization Details Summary</h1>
                    <h2>Top {len(top_results_display)} Parameter Sets</h2>
                    <p>Sorted by Profit Factor (desc), then Total PnL (desc)</p>
                    {top_results_display.to_html(index=False, escape=False)}
                    
                    {plot_img_tag}  <!-- Embed plot image here -->
                    
                </body>
                </html>
                """
                with open(report_filepath, "w") as f:
                    f.write(html_content)
                logger.info(
                    f"Saved optimization details summary HTML report to {report_filepath}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to generate optimization details HTML report: {e}",
                    exc_info=True,
                )

    # Save Aggregated Summary CSV (Applies to both modes, results_df is ready)
    summary_file_path = Path(args.summary_file)
    summary_file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        results_df.to_csv(summary_file_path, index=False)
        logger.info(f"Saved aggregated analysis summary to {summary_file_path}")
    except Exception as e:
        logger.error(f"Failed to save aggregated summary: {e}")

    # Generate Overall Plots (Only possible if we have combined_trades_df_for_plotting)
    # --- This call is now handled earlier to capture the plot path ---
    # if combined_trades_df_for_plotting is not None and not analyze_details_mode:
    #     plot_overall_results(
    #         combined_trades_df_for_plotting, results_df, top_results, args
    #     )
    # elif args.plotting and analyze_details_mode:
    #     # Plotting for details mode (just the top N) is handled earlier now
    #     pass
    # elif args.plotting and combined_trades_df_for_plotting is None:
    #      logger.warning(
    #         "Overall plots cannot be generated as raw trade data is not loaded."
    #     )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Trade CSV files or Optimization Detail files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/optimization"),
        help="Directory containing the trade log CSV files (usually ./results/optimization/trades or ./results/optimization for details).",
    )
    parser.add_argument(
        "--analyze-details",
        action="store_true",
        help="Analyze optimization detail files (*_optimize_details_*.csv) instead of trade logs (*_trades.csv).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Filter analysis to a specific strategy name.",
    )
    parser.add_argument(
        "--symbol", type=str, default=None, help="Filter analysis to a specific symbol."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top performing parameter sets to display in console log. Default: 10",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. Default: INFO",
    )
    # Output arguments
    parser.add_argument(
        "--summary-file",
        type=str,
        default="results/analysis/summary/analysis_summary.csv",
        help="Path to save the aggregated summary CSV file. Default: results/analysis/summary/analysis_summary.csv",
    )
    parser.add_argument(
        "--plotting",
        action="store_true",
        help="Enable generation of performance plots and HTML reports.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="results/analysis/plots",
        help="Directory to save generated plots. Default: results/analysis/plots",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="results/analysis/reports",
        help="Directory to save generated HTML reports. Default: results/analysis/reports",
    )

    args = parser.parse_args()

    # Set logging level
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)  # Apply to root logger
    logger.setLevel(log_level)  # Apply to this script's logger too

    results_path = Path(args.results_dir)
    if not results_path.is_dir():
        logger.error(f"Results directory not found: {args.results_dir}")
        return

    # 1. Find relevant files (conditional based on flag)
    files_meta = find_files(
        results_path, args.strategy, args.symbol, args.analyze_details
    )

    # 2. Load and process data (conditional based on flag)
    combined_data_df = load_and_process_data(files_meta, args.analyze_details)

    # 3. Analyze, Plot, Report, and Save Results
    if combined_data_df is not None:
        analyze_trades(combined_data_df, top_n=args.top_n, args=args)
    else:
        logger.error("Analysis aborted due to issues loading data.")


if __name__ == "__main__":
    main()
    logger.info("--- Trade analysis script finished. ---")
