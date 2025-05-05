import pandas as pd
from pathlib import Path
import logging
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any, Optional
import numpy as np

# Setup logger
logger = logging.getLogger(__name__)


# --- Plotting Function ---
def plot_equity_curve(
    equity_curve: pd.Series, plot_filename: Path, title_prefix: str = "Portfolio"
):
    """Generates and saves an equity curve plot."""
    if equity_curve is None or equity_curve.empty:
        logger.warning(
            f"Equity curve data empty or None for {plot_filename.stem}. Skipping plot."
        )
        return False
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity_curve.index, equity_curve.values, label="Equity Curve")

        # Formatting the plot
        ax.set_title(f"{title_prefix} Equity Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
        fig.autofmt_xdate()  # Auto format date labels
        ax.legend()

        plot_filename.parent.mkdir(parents=True, exist_ok=True)  # Ensure dir exists
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close(fig)  # Close the figure to free memory
        logger.info(f"Equity curve plot saved to: {plot_filename}")
        return True
    except Exception as e:
        logger.error(
            f"Error generating or saving plot {plot_filename}: {e}", exc_info=True
        )
        return False


def plot_rolling_metric(
    pnl_series: pd.Series,
    window: int,
    plot_filename: Path,
    metric_name: str = "Sharpe Ratio (Trade PnL)",
    title_prefix: str = "Rolling Metric",
):
    """Generates and saves a rolling Sharpe ratio (or similar mean/std metric) plot based on PnL series."""
    if pnl_series is None or pnl_series.empty or len(pnl_series) < window:
        logger.warning(
            f"PnL series too short ({len(pnl_series)}<{window}) or empty for rolling {metric_name} plot ({plot_filename.stem}). Skipping."
        )
        return False
    try:
        # Calculate rolling mean and std dev of PnL
        rolling_mean = pnl_series.rolling(window=window, min_periods=window).mean()
        rolling_std = pnl_series.rolling(window=window, min_periods=window).std()

        # Calculate rolling Sharpe (handle division by zero)
        rolling_metric = rolling_mean / rolling_std.replace(0, np.nan)
        rolling_metric.dropna(inplace=True)  # Drop initial NaNs

        if rolling_metric.empty:
            logger.warning(
                f"Rolling {metric_name} calculation resulted in empty series. Skipping plot."
            )
            return False

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            rolling_metric.index,
            rolling_metric.values,
            label=f"Rolling {metric_name} ({window}-trade)",
        )

        # Formatting
        ax.set_title(f"{title_prefix}: Rolling {metric_name} Over Trades")
        ax.set_xlabel("Trade Number")
        ax.set_ylabel(metric_name)
        ax.grid(True)
        ax.legend()

        plot_filename.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close(fig)
        logger.info(f"Rolling {metric_name} plot saved to: {plot_filename}")
        return True
    except Exception as e:
        logger.error(
            f"Error generating or saving rolling {metric_name} plot {plot_filename}: {e}",
            exc_info=True,
        )
        return False


# --- HTML Report Function ---
def generate_html_report(
    metrics: Dict[str, Any],
    plot_path: Optional[Path],  # Path to the plot image (e.g., from backtrader)
    report_filename: Path,
    report_title: str,
    test_params: Dict[str, Any],
    strategy_params: Dict[str, Any],
    trade_list_path: Optional[Path] = None,  # Added path for trade list CSV
    # Removed optimization_plot_paths and config_details (handle details in title/params)
):
    """Generates an HTML report with metrics, parameters, plot, and optionally trades."""
    try:
        # Embed plot image as base64 if path is provided
        img_tag = "<p><i>Plot generation failed or was skipped.</i></p>"
        if plot_path and plot_path.is_file():
            try:
                with open(plot_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                # Added title attribute for accessibility/hover
                img_tag = f'<img src="data:image/png;base64,{encoded_string}" alt="Backtest Chart" title="Backtest Chart" style="max-width: 100%; height: auto;">'
            except Exception as img_e:
                logger.error(f"Error embedding plot image {plot_path}: {img_e}")
                img_tag = f"<p><i>Error embedding plot image: {img_e}</i></p>"
        elif plot_path:
            img_tag = f"<p><i>Plot file not found: {plot_path}</i></p>"

        # --- Build Parameters Table ---
        # Use test_params directly as they are already structured
        params_html = "<table>\n<tr><th>Parameter Type</th><th>Parameter</th><th>Value</th></tr>\n"
        general_param_count = len(test_params)
        if general_param_count > 0:
            first = True
            for key, value in test_params.items():
                row_span_attr = f' rowspan="{general_param_count}"' if first else ""
                section_name = "Run Setup" if first else ""
                params_html += f"<tr><td{row_span_attr}>{section_name}</td><td>{key}</td><td>{value}</td></tr>\n"
                first = False

        strat_param_count = len(strategy_params)
        if strat_param_count > 0:
            first = True
            for key, value in strategy_params.items():
                # Nicely format tuples, None, etc.
                if value is None:
                    value_str = "<i>None</i>"
                elif isinstance(value, tuple):
                    value_str = f"({value[0]}, {value[1]})"
                else:
                    value_str = str(value)

                row_span_attr = f' rowspan="{strat_param_count}"' if first else ""
                section_name = "Strategy Params" if first else ""
                params_html += f"<tr><td{row_span_attr}>{section_name}</td><td>{key}</td><td>{value_str}</td></tr>\n"
                first = False
        else:
            params_html += '<tr><td>Strategy Params</td><td colspan="2"><i>None Loaded/Required</i></td></tr>\n'
        params_html += "</table>\n"

        # --- Build Performance Metrics Table ---
        perf_html = "<table>\n<tr><th>Metric</th><th>Value</th></tr>\n"
        # Use keys from metrics dict directly
        for key, value in metrics.items():
            # Format based on key name for percentages etc.
            metric_name = key.replace("_", " ").title()
            if "percent" in key or "rate" in key:
                value_str = (
                    f"{value:.2f}%" if isinstance(value, (int, float)) else str(value)
                )
            elif isinstance(value, float):
                value_str = f"{value:.4f}"  # Default float format
                if key == "profit_factor" and value == float("inf"):
                    value_str = "Inf"
                if key == "sharpe_ratio" or key == "sortino_ratio":
                    value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            perf_html += f"<tr><td>{metric_name}</td><td>{value_str}</td></tr>\n"
        perf_html += "</table>\n"

        # --- Build Trades Table ---
        trades_html = "<p><i>Trade list not provided or file not found.</i></p>"
        if trade_list_path and trade_list_path.is_file():
            try:
                trades_df = pd.read_csv(trade_list_path)
                trades_html = trades_df.to_html()
            except Exception as trades_e:
                logger.error(
                    f"Error reading trade list CSV {trade_list_path}: {trades_e}"
                )
                trades_html = "<p><i>Error reading trade list CSV.</i></p>"

        # --- Build Full HTML ---
        # Simplified title handling, assume report_title covers necessary context
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_title}</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 600px; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
                th {{ background-color: #f2f2f2; }}
                /* Adjusted image style */
                img {{ margin-top: 15px; border: 1px solid #ccc; max-width: 95%; height: auto; display: block; margin-left: auto; margin-right: auto;}}
            </style>
        </head>
        <body>
            <h1>{report_title}</h1>
            
            <h2>Parameters</h2>
            {params_html}
            
            <h2>Performance Metrics</h2>
            {perf_html} 
            
            <h2>Chart</h2>
            {img_tag}
            
            <h2>Trades</h2>
            {trades_html}

            <p><i>Note: Metrics calculated by backtrader analyzers. Chart generated by backtrader. Trade list from CSV.</i></p>
        </body>
        </html>
        """

        report_filename.parent.mkdir(parents=True, exist_ok=True)  # Ensure dir exists
        with open(report_filename, "w") as f:
            f.write(html_content)
        logger.info(f"HTML report saved to: {report_filename}")
        return True
    except Exception as e:
        logger.error(
            f"Error generating HTML report {report_filename}: {e}", exc_info=True
        )
        return False
