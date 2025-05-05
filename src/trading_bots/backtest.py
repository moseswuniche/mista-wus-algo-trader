# mypy: disable-error-code=operator
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, cast, Tuple, Union
import logging
from datetime import time, datetime  # Updated imports
import inspect
import backtrader as bt  # Import backtrader
from pathlib import Path
import os  # Added for directory creation

# --- Add type ignore for the whole file if necessary, or specific function ---
# type: ignore[operator]

from .strategies.base_strategy import BaseStrategy
from .technical_indicators import (
    calculate_atr,
    calculate_sma,
    calculate_ema,
)  # Import ATR function
from .strategies import (
    MovingAverageCrossoverStrategy,
    BollingerBandReversionStrategy,
    ScalpingStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    HybridStrategy,
)  # Import for type checking

# Import the configuration model
from .config_models import BacktestRunConfig, ValidationError

# Get logger instance
logger = logging.getLogger(__name__)

# --- Centralized Strategy Map (Align with optimize.py) ---
# Map strategy *class names* (matching config keys) to classes
STRATEGY_MAP = {
    "MovingAverageCrossoverStrategy": MovingAverageCrossoverStrategy,
    "ScalpingStrategy": ScalpingStrategy,
    "BollingerBandReversionStrategy": BollingerBandReversionStrategy,
    "MomentumStrategy": MomentumStrategy,
    "MeanReversionStrategy": MeanReversionStrategy,
    "BreakoutStrategy": BreakoutStrategy,
    "HybridStrategy": HybridStrategy,
}


# --- Helper: PandasData Feed with Standard Column Names ---
# Ensure datetime index is localized or naive consistently
class PandasDataFeed(bt.feeds.PandasData):
    lines = (
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    params = (
        ("datetime", None),  # Use index
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),  # Not used
    )


# --- Main Backtest Function (Rewritten for Backtrader) ---
def run_backtest(
    data: pd.DataFrame,
    config: BacktestRunConfig,
    # plot_filename: Optional[str] = None, # Removed: Use config.plot_path instead
) -> tuple[
    float, float, dict, dict, object | None
]:  # Return: final_value, max_drawdown, trade_analysis, all_analyzers, plot_object
    """
    Runs a backtest using the Backtrader engine.

    Args:
        data: DataFrame with historical OHLCV data (lowercase columns), index must be DatetimeIndex.
        config: Configuration object for the backtest run.
        # plot_filename: If provided, saves the backtrader plot to this path.

    Returns:
        A tuple containing:
        - Final portfolio value.
        - Maximum drawdown percentage.
        - Trade analysis dictionary (from TradeAnalyzer).
        - Dictionary containing all analyzer results.
        - Optional plot object if config.plot_path is set (plot is also saved to file).
        Or a default tuple (0.0, 0.0, {}, {}, None) on error.
    """
    logger.debug(
        f"--- Starting Backtrader Run for: {config.symbol} / {config.strategy_short_name} ---"  # Use short name from config
    )
    # logger.debug(f"Backtest Config: {config.model_dump()}") # Can be verbose

    # --- Input Validation & Data Preparation --- #
    if data.empty:
        logger.error("Input data is empty. Cannot run backtest.")
        return (0.0, 0.0, {}, {}, None)  # Return default tuple on error
    if not isinstance(data.index, pd.DatetimeIndex):
        # Attempt conversion if possible, otherwise error
        try:
            data.index = pd.to_datetime(data.index)
            logger.warning("Converted data index to DatetimeIndex.")
        except Exception as e:
            logger.error(f"Data index must be a DatetimeIndex, conversion failed: {e}")
            return (0.0, 0.0, {}, {}, None)  # Return default tuple on error

    # Ensure data index is timezone-naive or consistently localized (e.g., UTC)
    # Backtrader can be sensitive to timezone mixing.
    if data.index.tz is not None:
        logger.debug(f"Data index timezone: {data.index.tz}. Converting to UTC.")
        try:
            data.index = data.index.tz_convert("UTC")
        except Exception as e:
            logger.error(f"Failed to convert index to UTC: {e}")
            return (0.0, 0.0, {}, {}, None)  # Return default tuple on error
    else:
        logger.debug("Data index is timezone-naive.")

    # Rename columns for PandasDataFeed if they are not lowercase already
    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(set(data.columns)):
        logger.error(
            f"Dataframe missing required columns: {required_cols - set(data.columns)}"
        )
        return (0.0, 0.0, {}, {}, None)  # Return default tuple on error

    # --- Cerebro Engine Setup --- #
    cerebro = bt.Cerebro(stdstats=False, plot=False)  # Disable standard stats/plotting

    # --- Add Data Feed --- #
    # Ensure start/end dates align with the config if needed, otherwise use full data
    # start_date = pd.to_datetime(config.start_date)
    # end_date = pd.to_datetime(config.end_date)
    # data_feed = PandasDataFeed(dataname=data[start_date:end_date])
    data_feed = PandasDataFeed(dataname=data)
    cerebro.adddata(data_feed, name=config.symbol)

    # --- Instantiate Strategy --- #
    strategy_class = STRATEGY_MAP.get(
        config.strategy_short_name
    )  # Use short name from config
    if not strategy_class:
        logger.error(
            f"Strategy class name '{config.strategy_short_name}' not found in map."
        )
        return (0.0, 0.0, {}, {}, None)  # Return default tuple on error
    try:
        # Prepare parameters to pass to addstrategy
        # BaseStrategy handles extracting relevant params like SL/TP/Filters from its self.params
        strategy_params_to_pass = config.strategy_params.copy()
        # Removed checks for config.stop_loss_pct etc. - these are passed within strategy_params_to_pass
        # if config.stop_loss_pct is not None:
        #     strategy_params_to_pass['stop_loss_pct'] = config.stop_loss_pct
        # if config.take_profit_pct is not None:
        #     strategy_params_to_pass['take_profit_pct'] = config.take_profit_pct
        # if config.time_window is not None:
        #      strategy_params_to_pass['time_window'] = config.time_window
        # if config.liquidity_threshold is not None:
        #      strategy_params_to_pass['liquidity_threshold'] = config.liquidity_threshold

        logger.debug(
            f"Adding strategy {config.strategy_short_name} with params: {strategy_params_to_pass}"
        )
        cerebro.addstrategy(strategy_class, **strategy_params_to_pass)

    except Exception as e:
        logger.error(
            f"Error adding strategy '{config.strategy_short_name}' with params {strategy_params_to_pass}: {e}",
            exc_info=True,
        )
        return (0.0, 0.0, {}, {}, None)  # Return default tuple on error

    # --- Configure Broker --- #
    cerebro.broker.set_cash(config.initial_balance)
    commission_decimal = config.commission_bps / 10000.0
    cerebro.broker.setcommission(commission=commission_decimal)
    # Set slippage model if needed: cerebro.broker.set_slippage_perc(perc=0.001) # 0.1%

    # --- Add Analyzers --- #
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tradeanalyzer")
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,
        riskfreerate=0.0,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")
    # Add Plot Data Info analyzer if plotting
    # No need to add separate analyzer for plotting, cerebro.plot() handles it

    # --- Run Backtest --- #
    try:
        logger.info(
            f"Running Cerebro for {config.symbol} / {config.strategy_short_name}..."
        )
        results = cerebro.run()
        if not results or len(results) == 0:
            logger.warning("Cerebro run returned no results.")
            return (0.0, 0.0, {}, {}, None)  # Return default tuple on error
        logger.info("Cerebro run finished.")

        strat_instance = results[0]  # Get the first strategy instance
        analyzers = strat_instance.analyzers

    except Exception as e:
        logger.error(f"Exception during Cerebro run: {e}", exc_info=True)
        return (0.0, 0.0, {}, {}, None)  # Return default tuple on error

    # --- Extract and Format Results --- #
    try:
        metrics = {}
        # Basic Portfolio Stats
        metrics["initial_balance"] = config.initial_balance
        metrics["final_balance"] = cerebro.broker.getvalue()
        metrics["net_profit"] = metrics["final_balance"] - metrics["initial_balance"]
        metrics["net_profit_pct"] = (
            (metrics["net_profit"] / metrics["initial_balance"]) * 100
            if metrics["initial_balance"]
            else 0
        )

        # Returns Analyzer
        returns_analyzer = analyzers.returns.get_analysis()
        metrics["cumulative_return_pct"] = (
            returns_analyzer.get("rtot", 0) * 100
        )  # Total return rate
        # Note: 'rnorm100' often represents annualized return in backtrader examples
        metrics["annualized_return_pct"] = returns_analyzer.get("rnorm100", 0)

        # Sharpe Ratio Analyzer
        sharpe_analyzer = analyzers.sharpe.get_analysis()
        metrics["sharpe_ratio"] = sharpe_analyzer.get("sharperatio", 0)
        # Sometimes sharpe ratio might be None if std dev is 0
        if metrics["sharpe_ratio"] is None:
            metrics["sharpe_ratio"] = 0.0

        # Drawdown Analyzer
        drawdown_analyzer = analyzers.drawdown.get_analysis()
        metrics["max_drawdown_pct"] = drawdown_analyzer.max.drawdown
        metrics["max_drawdown_abs"] = drawdown_analyzer.max.moneydown
        metrics["longest_drawdown_duration"] = drawdown_analyzer.max.len

        # Trade Analyzer
        trade_analyzer = (
            analyzers.tradeanalyzer.get_analysis()
            if hasattr(analyzers, "tradeanalyzer")
            else {}
        )
        metrics["total_trades"] = trade_analyzer.get("total", {}).get("closed", 0)
        metrics["winning_trades"] = trade_analyzer.get("won", {}).get("total", 0)
        metrics["losing_trades"] = trade_analyzer.get("lost", {}).get("total", 0)
        metrics["win_rate"] = (
            (metrics["winning_trades"] / metrics["total_trades"] * 100)
            if metrics["total_trades"] > 0
            else 0
        )
        metrics["average_trade_pnl"] = (
            trade_analyzer.get("pnl", {}).get("net", {}).get("average", 0)
        )
        metrics["average_winning_trade"] = (
            trade_analyzer.get("won", {}).get("pnl", {}).get("average", 0)
        )
        metrics["average_losing_trade"] = (
            trade_analyzer.get("lost", {}).get("pnl", {}).get("average", 0)
        )
        metrics["max_consecutive_wins"] = (
            trade_analyzer.get("streak", {}).get("won", {}).get("longest", 0)
        )
        metrics["max_consecutive_losses"] = (
            trade_analyzer.get("streak", {}).get("lost", {}).get("longest", 0)
        )
        metrics["profit_factor"] = (
            (
                trade_analyzer.get("won", {}).get("pnl", {}).get("total", 0)
                / abs(trade_analyzer.get("lost", {}).get("pnl", {}).get("total", 0))
            )
            if trade_analyzer.get("lost", {}).get("pnl", {}).get("total", 0) != 0
            else (
                float("inf")
                if trade_analyzer.get("won", {}).get("pnl", {}).get("total", 0) > 0
                else 0
            )
        )

        # SQN Analyzer
        sqn_analyzer = analyzers.sqn.get_analysis()
        metrics["sqn"] = sqn_analyzer.get("sqn", 0)

        # Replace NaN/Inf with 0 or None for JSON compatibility if needed
        for k, v in metrics.items():
            if isinstance(v, (float, int)) and (np.isnan(v) or np.isinf(v)):
                metrics[k] = 0.0  # Replace with 0 for simplicity
            elif isinstance(v, pd.Timestamp):
                metrics[k] = v.isoformat()  # Convert timestamps

        logger.info(
            f"Backtest Complete. Final Balance: {metrics['final_balance']:.2f}, Sharpe: {metrics.get('sharpe_ratio', 0):.3f}"
        )

        # --- Save Trade List (if path provided in config) ---
        if config.trade_list_output_path:
            logger.info(f"Saving trade list to: {config.trade_list_output_path}")
            try:
                # Ensure output directory exists
                output_dir = Path(config.trade_list_output_path).parent
                os.makedirs(output_dir, exist_ok=True)

                trades_data = []
                # Check if the structure is as expected (can vary slightly with bt versions)
                if trade_analyzer and isinstance(trade_analyzer.get("trades"), dict):
                    for t_id, t_data in trade_analyzer["trades"].items():
                        # Defensive access to potentially missing keys
                        entry_dt = (
                            bt.num2date(t_data.get("dtopen")).isoformat()
                            if t_data.get("dtopen")
                            else None
                        )
                        exit_dt = (
                            bt.num2date(t_data.get("dtclose")).isoformat()
                            if t_data.get("dtclose")
                            else None
                        )
                        # Approx exit price calculation might need refinement based on actual data
                        exit_price = (
                            (t_data.get("pnlcomm", 0) / t_data.get("size", 1))
                            + t_data.get("price", 0)
                            if t_data.get("size")
                            else t_data.get("price", 0)
                        )

                        trades_data.append(
                            {
                                "trade_id": t_id,
                                "status": t_data.get("status"),
                                "ref": t_data.get("ref"),
                                "symbol": config.symbol,
                                "type": (
                                    "Long"
                                    if t_data.get("direction") == "long"
                                    else (
                                        "Short"
                                        if t_data.get("direction") == "short"
                                        else "Unknown"
                                    )
                                ),
                                "entry_dt": entry_dt,
                                "entry_price": t_data.get("price"),
                                "entry_comm": t_data.get("commissionopen"),
                                "exit_dt": exit_dt,
                                "exit_price": exit_price,
                                "exit_comm": t_data.get("commissionclose"),
                                "size": t_data.get("size"),
                                "pnl": t_data.get("pnl"),
                                "pnl_comm": t_data.get("pnlcomm"),
                                "value": t_data.get("value"),
                                "bar_open": t_data.get("baropen"),
                                "bar_close": t_data.get("barclose"),
                                "bar_len": t_data.get("barlen"),
                            }
                        )
                else:
                    logger.warning(
                        f"Trade Analyzer 'trades' data not found or not in expected dictionary format in trade_analyzer results: {trade_analyzer}"
                    )

                if trades_data:
                    trades_df = pd.DataFrame(trades_data)
                    trades_df.to_csv(config.trade_list_output_path, index=False)
                    logger.info(f"Successfully saved {len(trades_df)} trades.")
                else:
                    logger.warning("No trade data extracted to save.")
                    # Create empty file to signify attempt
                    Path(config.trade_list_output_path).touch()

            except Exception as e:
                logger.error(
                    f"Failed to save trade list to {config.trade_list_output_path}: {e}",
                    exc_info=True,
                )

        # --- Plotting (if path provided in config) ---
        plot_object = None
        if config.plot_path:
            logger.info(f"Generating and saving plot to: {config.plot_path}")
            try:
                # Ensure output directory exists
                plot_dir = Path(config.plot_path).parent
                os.makedirs(plot_dir, exist_ok=True)
                # Generate and save the plot, store the plot object
                plot_object = cerebro.plot(
                    style="candlestick",
                    barup="green",
                    bardown="red",
                    volume=True,
                    filename=config.plot_path,
                    savefig=True,
                )
                # Note: cerebro.plot returns a list of figures/axes lists
                # We might just return the fact that plotting occurred, or the main figure if needed.
                # Returning the object allows potential further manipulation but can be complex.
                # Let's keep it simple and return the object list from cerebro.plot for now.
                logger.info("Plot generated and saved successfully.")
            except ImportError:
                logger.warning(
                    "Plotting libraries (matplotlib) not installed. Cannot generate plot."
                )
            except Exception as e:
                logger.error(
                    f"Error generating or saving backtrader plot to {config.plot_path}: {e}",
                    exc_info=True,
                )

        # --- Return Results --- #
        all_analyzers_dict = {
            name: analyzer.get_analysis() for name, analyzer in analyzers.items()
        }

        # Return key metrics and the plot object
        return (
            metrics.get("final_balance", 0.0),
            metrics.get("max_drawdown_pct", 0.0),
            trade_analyzer,
            all_analyzers_dict,
            plot_object,
        )

    except KeyError as e:
        logger.error(
            f"Missing key in extracted metrics or analyzers: {e}", exc_info=True
        )
        return (0.0, 0.0, {}, {}, None)  # Return default tuple on error
    except Exception as e:
        logger.error(
            f"Error extracting metrics from Cerebro results: {e}", exc_info=True
        )
        return (0.0, 0.0, {}, {}, None)  # Return default tuple on error


# --- Remove old helper/metric functions not used by backtrader run ---
# (calculate_sharpe_ratio, calculate_max_drawdown, calculate_profit_factor were specific to vectorized)
# (parse_trading_hours might still be useful if filters applied outside strategy)


# --- Example Usage (Optional) ---
# if __name__ == '__main__':
#     # Create dummy data
#     dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
#     data = pd.DataFrame({
#         'open': np.random.rand(200) * 100 + 1000,
#         'high': np.random.rand(200) * 10 + 1000,
#         'low': np.random.rand(200) * -10 + 1000,
#         'close': np.random.rand(200) * 100 + 1000,
#         'volume': np.random.rand(200) * 10000 + 50000,
#     }, index=dates)
#     data['high'] = data[['open', 'close']].max(axis=1) + np.random.rand(200) * 5
#     data['low'] = data[['open', 'close']].min(axis=1) - np.random.rand(200) * 5

#     # Create dummy config
#     try:
#         config_dict = {
#             'symbol': 'DUMMY/USD',
#             'initial_balance': 10000,
#             'commission_bps': 5,
#             'units': 1.0,
#             'strategy_class_name': 'MovingAverageCrossoverStrategy',
#             'strategy_params': {'fast_period': 10, 'slow_period': 30, 'ma_type': 'EMA'},
#             'stop_loss_pct': 0.02,
#             'take_profit_pct': 0.04
#         }
#         config = BacktestRunConfig(**config_dict)
#     except ValidationError as e:
#         print(f"Config validation error: {e}")
#         exit()

#     # Configure logging
#     logging.basicConfig(level=logging.DEBUG,
#                         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#     # Run backtest
#     results = run_backtest(data=data, config=config)

#     if results:
#         print("\n--- Backtest Results ---")
#         for key, value in results.items():
#             print(f"{key}: {value}")
#     else:
#         print("Backtest failed.")
