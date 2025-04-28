import pandas as pd
import numpy as np
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, cast
import logging
import datetime # Added for filename generation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64
from io import BytesIO

# Assuming strategies are accessible via this import path
from .strategies import (
    Strategy,
    LongShortStrategy,
    MovingAverageCrossoverStrategy,
    RsiMeanReversionStrategy,
    BollingerBandReversionStrategy
)
from .backtest import run_backtest
from .data_utils import load_csv_data

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map strategy short names (used in args) to classes
STRATEGY_CLASS_MAP = {
    "LongShort": LongShortStrategy,
    "MACross": MovingAverageCrossoverStrategy,
    "RSIReversion": RsiMeanReversionStrategy,
    "BBReversion": BollingerBandReversionStrategy
}

# --- Parameter Loading Function (Similar to optimize.py but for specific values) ---
def load_best_params_from_config(config_path: str, symbol: str, strategy_class_name: str) -> Optional[Dict[str, Any]]:
    """Loads the best parameters for a given symbol and strategy from the optimization config file."""
    config_file = Path(config_path)
    if not config_file.is_file():
        logger.error(f"Parameter config file not found: {config_path}")
        return None

    try:
        with open(config_file, 'r') as f:
            all_configs = yaml.safe_load(f)
            if not isinstance(all_configs, dict):
                logger.error(f"Invalid format in {config_path}. Expected top-level dictionary.")
                return None
                
            symbol_configs = all_configs.get(symbol)
            if not isinstance(symbol_configs, dict):
                logger.error(f"Symbol {symbol} not found or invalid format in {config_path}.")
                return None
                
            strategy_params = symbol_configs.get(strategy_class_name)
            if not isinstance(strategy_params, dict):
                logger.error(f"Strategy {strategy_class_name} not found for symbol {symbol} in {config_path}.")
                return None
    except FileNotFoundError:
        logger.error(f"Parameter config file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading parameter config {config_path}: {e}", exc_info=True)
        return None
    
    # --- Adjust for LongShortStrategy's tuple parameters --- 
    if strategy_class_name == "LongShortStrategy":
        # Check if the split keys exist from the optimization save
        if 'return_thresh_low' in strategy_params and 'return_thresh_high' in strategy_params:
            strategy_params['return_thresh'] = (
                strategy_params.pop('return_thresh_low'), 
                strategy_params.pop('return_thresh_high')
            )
        elif 'return_thresh' not in strategy_params:
             logger.error(f"Missing return threshold parameters for LongShortStrategy in {config_path}")
             return None # Or raise error
             
        if 'volume_thresh_low' in strategy_params and 'volume_thresh_high' in strategy_params:
            strategy_params['volume_thresh'] = (
                strategy_params.pop('volume_thresh_low'), 
                strategy_params.pop('volume_thresh_high')
            )
        elif 'volume_thresh' not in strategy_params:
             logger.error(f"Missing volume threshold parameters for LongShortStrategy in {config_path}")
             return None # Or raise error
             
    # Ensure mypy knows the final type after modifications
    return cast(Optional[Dict[str, Any]], strategy_params)

# --- Function to create results directory ---
def ensure_results_dir(base_path: str) -> Path:
    results_path = Path(base_path)
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path

# --- Plotting Function ---
def plot_equity_curve(equity_curve: pd.Series, plot_filename: Path):
    """Generates and saves an equity curve plot."""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity_curve.index, equity_curve.values, label='Equity Curve')
        
        # Formatting the plot
        ax.set_title('Portfolio Equity Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
        fig.autofmt_xdate() # Auto format date labels
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close(fig) # Close the figure to free memory
        logger.info(f"Equity curve plot saved to: {plot_filename}")
        return True
    except Exception as e:
        logger.error(f"Error generating or saving plot {plot_filename}: {e}", exc_info=True)
        return False

# --- HTML Report Function ---
def generate_html_report(
    metrics: Dict[str, Any], 
    plot_path: Optional[Path], # Allow None if plot fails
    report_filename: Path, 
    test_params: Dict[str, Any], 
    strategy_params: Dict[str, Any]
):
    """Generates an HTML report with metrics, test parameters, and embedded plot."""
    try:
        # Embed plot image as base64 if path is provided
        img_tag = "<p><i>Plot generation failed or was skipped.</i></p>"
        if plot_path and plot_path.is_file():
            try:
                with open(plot_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                img_tag = f'<img src="data:image/png;base64,{encoded_string}" alt="Equity Curve" style="max-width: 100%; height: auto;">'
            except Exception as img_e:
                 logger.error(f"Error embedding plot image {plot_path}: {img_e}")
                 img_tag = f"<p><i>Error embedding plot image: {img_e}</i></p>"
        elif plot_path:
             img_tag = f"<p><i>Plot file not found: {plot_path}</i></p>"
        
        # --- Build Parameters Table ---
        params_html = "<table>\n<tr><th>Parameter Type</th><th>Parameter</th><th>Value</th></tr>\n"
        # General Test Params
        params_html += "<tr><td rowspan=\"6\">Test Setup</td><td>Strategy</td><td>{}</td></tr>\n".format(test_params.get('strategy','N/A'))
        params_html += "<tr><td>Symbol</td><td>{}</td></tr>\n".format(test_params.get('symbol','N/A'))
        params_html += "<tr><td>Forward Start</td><td>{}</td></tr>\n".format(test_params.get('fwd_start','N/A'))
        params_html += "<tr><td>Forward End</td><td>{}</td></tr>\n".format(test_params.get('fwd_end','N/A'))
        params_html += "<tr><td>Trade Units</td><td>{}</td></tr>\n".format(test_params.get('units','N/A'))
        params_html += "<tr><td>Commission (bps)</td><td>{}</td></tr>\n".format(test_params.get('commission_bps','N/A'))
        
        # Strategy Specific Params
        strat_param_count = len(strategy_params)
        if strat_param_count > 0:
            first = True
            for key, value in strategy_params.items():
                # Nicely format tuples
                if isinstance(value, tuple):
                    value_str = f"({value[0]}, {value[1]})"
                else:
                    value_str = str(value)
                    
                if first:
                    params_html += f'<tr><td rowspan=\"{strat_param_count}\">Strategy Params</td><td>{key}</td><td>{value_str}</td></tr>\n'
                    first = False
                else:
                    params_html += f'<tr><td>{key}</td><td>{value_str}</td></tr>\n'
        else:
             params_html += "<tr><td>Strategy Params</td><td colspan=\"2\"><i>None Loaded/Required</i></td></tr>\n"
        params_html += "</table>\n"
        
        # --- Build Performance Metrics Table --- 
        perf_html = "<table>\n<tr><th>Metric</th><th>Value</th></tr>\n"
        perf_html += f"<tr><td>Cumulative Profit</td><td>{metrics.get('cumulative_profit', 'N/A')}</td></tr>\n"
        perf_html += f"<tr><td>Final Balance</td><td>{metrics.get('final_balance', 'N/A')}</td></tr>\n"
        perf_html += f"<tr><td>Total Trades</td><td>{metrics.get('total_trades', 'N/A')}</td></tr>\n"
        perf_html += f"<tr><td>Win Rate</td><td>{metrics.get('win_rate_percent', 'N/A')}</td></tr>\n"
        # Add new metrics
        perf_html += f"<tr><td>Sharpe Ratio</td><td>{metrics.get('sharpe_ratio', 'N/A')}</td></tr>\n"
        perf_html += f"<tr><td>Max Drawdown</td><td>{metrics.get('max_drawdown_percent', 'N/A')}</td></tr>\n"
        perf_html += f"<tr><td>Profit Factor</td><td>{metrics.get('profit_factor', 'N/A')}</td></tr>\n"
        perf_html += "</table>\n"
        
        # --- Build Full HTML ---
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forward Test Report: {report_filename.stem}</title>
            <style>
                body {{ font-family: sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 500px; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
                th {{ background-color: #f2f2f2; }}
                img {{ margin-top: 15px; border: 1px solid #ccc; }}
            </style>
        </head>
        <body>
            <h1>Forward Test Report</h1>
            
            <h2>Test Parameters</h2>
            {params_html}
            
            <h2>Performance Metrics</h2>
            {perf_html} 
            
            <h2>Equity Curve</h2>
            {img_tag}
            
            <p><i>Detailed trade log saved separately in the 'trades' directory.</i></p>
        </body>
        </html>
        """
        
        with open(report_filename, 'w') as f:
            f.write(html_content)
        logger.info(f"HTML report saved to: {report_filename}")
        return True
    except Exception as e:
        logger.error(f"Error generating HTML report {report_filename}: {e}", exc_info=True)
        return False

# --- Main Forward Testing Function ---
def run_forward_test(
    strategy_short_name: str,
    param_config_path: str,
    symbol: str,
    data: pd.DataFrame,
    fwd_test_start: str,
    fwd_test_end: Optional[str],
    trade_units: float,
    commission_bps: float,
    initial_balance: float = 10000.0 # Add initial balance
):
    """Runs a backtest on a specified forward-testing period using pre-defined parameters."""
    logger.info(f"--- Starting Forward Test for {strategy_short_name} on {symbol} --- ")
    logger.info(f"Parameters loading from: {param_config_path}")
    fwd_end_str = fwd_test_end or "EndOfData" # For filename and report
    logger.info(f"Forward Test Period: {fwd_test_start} to {fwd_end_str}")
    logger.info(f"Initial Balance: {initial_balance}") # Log initial balance

    # 1. Get Strategy Class
    if strategy_short_name not in STRATEGY_CLASS_MAP:
        logger.error(f"Unknown strategy name: {strategy_short_name}. Available: {list(STRATEGY_CLASS_MAP.keys())}")
        return
    strategy_class = STRATEGY_CLASS_MAP[strategy_short_name]
    strategy_class_name = strategy_class.__name__

    # 2. Load Best Parameters
    best_params = load_best_params_from_config(param_config_path, symbol, strategy_class_name)
    if best_params is None:
        logger.error(f"Could not load best parameters for {symbol}/{strategy_class_name}. Aborting forward test.")
        return
        
    # Separate strategy-specific params from SL/TP/TSL params
    strategy_init_params = best_params.copy()
    sl_pct = strategy_init_params.pop('stop_loss_pct', None)
    tp_pct = strategy_init_params.pop('take_profit_pct', None)
    tsl_pct = strategy_init_params.pop('trailing_stop_loss_pct', None)
    # Keep a copy of *all* loaded params (incl SL/TP/TSL) for the report
    loaded_strategy_params_for_report = best_params.copy() 
        
    # 3. Instantiate Strategy
    try:
        # Use only the strategy-specific params for instantiation
        strategy_instance = strategy_class(**strategy_init_params)
        logger.info(f"Instantiated strategy {strategy_class_name} with core parameters: {strategy_init_params}")
        logger.info(f"Using SL={sl_pct}, TP={tp_pct}, TSL={tsl_pct} for backtest.")
    except Exception as e:
        logger.error(f"Error instantiating strategy {strategy_class_name} with params {strategy_init_params}: {e}", exc_info=True)
        return

    # 4. Slice Data for Forward Test Period
    try:
        fwd_start_dt = pd.to_datetime(fwd_test_start)
        if fwd_test_end:
            fwd_end_dt = pd.to_datetime(fwd_test_end)
            forward_data = data.loc[fwd_start_dt:fwd_end_dt].copy()
        else:
            forward_data = data.loc[fwd_start_dt:].copy()
            
        if forward_data.empty:
             logger.error(f"No data found for the specified forward test period ({fwd_test_start} to {fwd_test_end or 'end'}).")
             return
             
        logger.info(f"Data sliced for forward test: {len(forward_data)} rows from {forward_data.index.min()} to {forward_data.index.max()}")
    except Exception as e:
        logger.error(f"Error slicing data for forward test period ({fwd_test_start} - {fwd_test_end}): {e}", exc_info=True)
        return

    # 5. Run Backtest on Forward Data
    logger.info("Running backtest on the forward test data slice...")
    fwd_results = None 
    try:
        fwd_results = run_backtest(
            data=forward_data,
            strategy=strategy_instance,
            units=trade_units,
            initial_balance=initial_balance, # Pass initial balance
            commission_bps=commission_bps,
            stop_loss_pct=sl_pct, # Pass SL
            take_profit_pct=tp_pct, # Pass TP
            trailing_stop_loss_pct=tsl_pct # Pass TSL
        )
    except Exception as e:
        logger.error(f"Error running backtest during forward test: {e}", exc_info=True)
        
    # 6. Process and Save Results
    if fwd_results:
        logger.info("--- Forward Test Results --- ")
        # Extract metrics (including new ones)
        metrics = {
            'cumulative_profit': fwd_results.get('cumulative_profit'),
            'final_balance': fwd_results.get('final_balance'),
            'total_trades': fwd_results.get('total_trades'),
            'win_rate': fwd_results.get('win_rate'),
            'sharpe_ratio': fwd_results.get('sharpe_ratio'),
            'max_drawdown': fwd_results.get('max_drawdown'),
            'profit_factor': fwd_results.get('profit_factor'),
        }
        equity_curve = fwd_results.get('equity_curve')
        
        # Format metrics for better readability
        formatted_metrics = {
            'cumulative_profit': f"{metrics.get('cumulative_profit', 0):.4f}",
            'final_balance': f"{metrics.get('final_balance', 0):.2f}",
            'total_trades': metrics.get('total_trades', 0),
            'win_rate_percent': f"{metrics.get('win_rate', 0):.2f}%",
            'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'max_drawdown_percent': f"{metrics.get('max_drawdown', 0)*100:.2f}%",
            'profit_factor': f"{metrics.get('profit_factor', 0):.2f}"
        }
        logger.info(f"Cumulative Profit: {formatted_metrics['cumulative_profit']}")
        logger.info(f"Final Balance: {formatted_metrics['final_balance']}")
        logger.info(f"Total Trades: {formatted_metrics['total_trades']}")
        logger.info(f"Win Rate: {formatted_metrics['win_rate_percent']}")
        # Log new metrics
        logger.info(f"Sharpe Ratio: {formatted_metrics['sharpe_ratio']}")
        logger.info(f"Max Drawdown: {formatted_metrics['max_drawdown_percent']}")
        logger.info(f"Profit Factor: {formatted_metrics['profit_factor']}")

        # Prepare for saving
        base_results_dir = Path("results/forward_test")
        plots_dir = ensure_results_dir(str(base_results_dir / "plots"))
        reports_dir = ensure_results_dir(str(base_results_dir / "reports"))
        trades_dir = ensure_results_dir(str(base_results_dir / "trades"))
        
        # Sanitize dates for filename
        fwd_start_fn = fwd_test_start.replace('-', '')
        fwd_end_fn = fwd_end_str.replace('-', '') # Use the string version which handles None
        base_filename = f"{strategy_short_name}_{symbol}_{fwd_start_fn}_{fwd_end_fn}"

        # Generate and Save Plot
        plot_generated = False
        plot_filename = plots_dir / f"{base_filename}_equity.png"
        if isinstance(equity_curve, pd.Series) and not equity_curve.empty:
            plot_generated = plot_equity_curve(equity_curve, plot_filename)
        else:
             logger.warning("Equity curve data not available or empty, skipping plot generation.")
             
        # Gather test parameters for the report
        test_run_params = {
            'strategy': strategy_short_name,
            'symbol': symbol,
            'fwd_start': fwd_test_start,
            'fwd_end': fwd_end_str,
            'units': trade_units,
            'commission_bps': commission_bps,
            'initial_balance': initial_balance # Add initial balance to report params
        }
        
        # Generate and Save HTML Report
        report_filename = reports_dir / f"{base_filename}_report.html"
        generate_html_report(
            metrics=formatted_metrics, # Pass formatted metrics including new ones
            plot_path=plot_filename if plot_generated else None, 
            report_filename=report_filename,
            test_params=test_run_params,
            # Pass the *full* original set of loaded params including SL/TP/TSL
            strategy_params=loaded_strategy_params_for_report 
        )

        # Save trade summary to CSV
        summary_df = fwd_results.get('performance_summary')
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            logger.info("\n--- Forward Test Trade Summary ---")
            logger.info(f"\n{summary_df.to_string()}")
            trades_filename = trades_dir / f"{base_filename}_trades.csv"
            try:
                summary_df.to_csv(trades_filename)
                logger.info(f"Forward test trades saved to: {trades_filename}")
            except Exception as e:
                logger.error(f"Error saving trades CSV to {trades_filename}: {e}", exc_info=True)
        else:
            logger.info("No trades executed during the forward test period. No trades CSV saved.")
            
    else: 
        logger.warning("Backtest function returned no results. No results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Simulated Forward Test using Optimized Parameters")
    parser.add_argument("-s", "--strategy", type=str, required=True,
                        choices=list(STRATEGY_CLASS_MAP.keys()),
                        help="Short name of the strategy previously optimized.")
    parser.add_argument("--symbol", type=str, required=True,
                        help="Trading symbol (e.g., BTCUSDT) to load data and parameters for.")
    parser.add_argument("--param-config", type=str, default="config/best_params.yaml",
                        help="Path to the YAML file containing the best parameters.")
    parser.add_argument("-f", "--file", type=str, default=None,
                        help="Path to the FULL historical data CSV file. If None, attempts default path (e.g., data/SYMBOL_1d.csv).")
    parser.add_argument("--fwd-start", type=str, required=True,
                        help="Start date (YYYY-MM-DD) for the forward testing period.")
    parser.add_argument("--fwd-end", type=str, default=None,
                        help="Optional end date (YYYY-MM-DD) for the forward testing period. If None, uses data until the end.")
    parser.add_argument("-u", "--units", type=float, default=1.0,
                        help="Amount/Units of asset to trade.")
    parser.add_argument("--commission", type=float, default=10.0,
                        help="Commission fee in basis points (e.g., 10 for 0.1%). Default: 10.")
    parser.add_argument("--balance", type=float, default=10000.0, 
                        help="Initial balance for the forward test backtest. Default: 10000.0")
    parser.add_argument("--log", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level. Default: INFO")

    args = parser.parse_args()

    # Set logging level
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    # Determine data file path (using daily '1d' as a common default if not specified)
    if args.file is None:
        args.file = f"data/{args.symbol}_1d.csv" # Assuming daily data is common for fwd test
        logger.info(f"Data file not specified, attempting default: {args.file}")
        # Add fallback for bitcoin.csv if needed?
        fallback_file = "src/trading_bots/bitcoin.csv" # Assuming this is daily BTC
        if not Path(args.file).is_file() and args.symbol == "BTCUSDT" and Path(fallback_file).is_file():
             args.file = fallback_file
             logger.info(f"Default file not found, using fallback: {args.file}")
             
    try:
        logger.info(f"Starting forward test run with args: {args}")
        # Load FULL dataset
        full_data = load_csv_data(args.file, symbol=args.symbol)
        full_data.sort_index(inplace=True)
        logger.info(f"Loaded full dataset: {len(full_data)} rows from {full_data.index.min()} to {full_data.index.max()}")

        # Run the forward test function
        run_forward_test(
            strategy_short_name=args.strategy,
            param_config_path=args.param_config,
            symbol=args.symbol,
            data=full_data, 
            fwd_test_start=args.fwd_start,
            fwd_test_end=args.fwd_end,
            trade_units=args.units,
            commission_bps=args.commission,
            initial_balance=args.balance # Pass balance from args
        )

    except FileNotFoundError as fnf:
        logger.error(f"File Not Found: {fnf}", exc_info=True)
    except ValueError as ve:
        logger.error(f"Configuration or Value Error: {ve}", exc_info=True)
    except IOError as ioe:
        logger.error(f"File Reading Error: {ioe}", exc_info=True)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)

    logger.info("--- Forward test script finished. ---") 