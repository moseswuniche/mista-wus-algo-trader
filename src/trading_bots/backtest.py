import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, cast, Tuple
import logging
from datetime import time  # For Seasonality

from .strategies.base_strategy import Strategy
from .technical_indicators import calculate_atr  # Import ATR function
from .strategies import LongShortStrategy  # Import for type checking

# Get logger instance
logger = logging.getLogger(__name__)


# --- Helper: Parse Time String ---
def parse_trading_hours(hours_str: Optional[str]) -> Optional[Tuple[int, int]]:
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


# --- Performance Metrics Calculations ---
def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 365
) -> float:
    """Calculates the Sharpe ratio from a pandas Series of returns."""
    if returns.empty:
        return 0.0
    std_dev = returns.std()
    if std_dev == 0 or pd.isna(std_dev):  # Handle zero or NaN standard deviation
        # If mean return is positive, Sharpe is inf, otherwise 0? Convention varies.
        # Returning 0 is safer and avoids potential inf issues downstream.
        return 0.0
    excess_returns = returns - risk_free_rate / periods_per_year
    # Cast result to float to satisfy mypy
    return cast(
        float,
        (excess_returns.mean() / std_dev) * np.sqrt(periods_per_year),
    )


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculates the maximum drawdown from an equity curve."""
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()
    # Cast result to float to satisfy mypy
    return cast(float, abs(max_drawdown))  # Return as a positive percentage


def calculate_profit_factor(performance_summary: pd.DataFrame) -> float:
    """Calculates the profit factor from the performance summary DataFrame."""
    if performance_summary.empty or "Profit/Loss" not in performance_summary.columns:
        return 0.0
    gross_profits = performance_summary[performance_summary["Profit/Loss"] > 0][
        "Profit/Loss"
    ].sum()
    gross_losses = abs(
        performance_summary[performance_summary["Profit/Loss"] < 0]["Profit/Loss"].sum()
    )
    if gross_losses == 0:
        return float("inf") if gross_profits > 0 else 0.0  # Avoid division by zero
    # Cast result to float to satisfy mypy
    return cast(float, gross_profits / gross_losses)


# --- Main Backtest Function ---
def run_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    symbol: str,
    units: float,
    initial_balance: float = 10000.0,
    commission_bps: float = 0.0,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    trailing_stop_loss_pct: Optional[float] = None,
    # --- Universal Filters ---
    apply_atr_filter: bool = False,
    atr_filter_period: int = 14,  # Default period if filter applied
    atr_filter_multiplier: float = 1.5,
    atr_filter_sma_period: int = 100,
    apply_seasonality_filter: bool = False,
    allowed_trading_hours_utc: Optional[str] = None,  # e.g., '5-17'
    apply_seasonality_to_symbols: Optional[str] = None,  # e.g., 'SOLUSDT,SUIUSDT'
) -> Dict[str, Any]:
    """
    Runs a bar-by-bar backtest simulating intra-bar SL/TP/TSL checks and applying filters.

    Args:
        data: DataFrame with historical OHLCV data, indexed by Date.
              Must include 'Open', 'High', 'Low', 'Close'.
              Must also contain 'atr' and 'atr_sma' columns if apply_atr_filter is True.
        strategy: An instantiated strategy object.
        symbol: The symbol being backtested (e.g., 'BTCUSDT').
        units: The fixed amount of the asset to trade per signal.
        initial_balance: Starting balance for calculating returns.
        commission_bps: Commission fee in basis points (e.g., 10 for 0.1%).
        stop_loss_pct: Optional fixed stop loss percentage (e.g., 0.02 for 2%).
        take_profit_pct: Optional fixed take profit percentage (e.g., 0.04 for 4%).
        trailing_stop_loss_pct: Optional trailing stop loss percentage (e.g., 0.01 for 1%).
        apply_atr_filter: If True, enables the ATR volatility filter.
        atr_filter_period: Period used for ATR calculation (for logging/info only).
        atr_filter_multiplier: Multiplier for ATR volatility filter threshold.
        atr_filter_sma_period: SMA period for ATR threshold baseline.
        apply_seasonality_filter: If True, enables the trading hours filter.
        allowed_trading_hours_utc: String like '5-17' for allowed UTC hours.
        apply_seasonality_to_symbols: Comma-separated string of symbols for seasonality filter.

    Returns:
        A dictionary containing backtest results.
    """
    required_cols = ["Open", "High", "Low", "Close"]
    if apply_atr_filter:
        required_cols.extend(["atr"])
        if atr_filter_sma_period > 0:
            required_cols.extend(["atr_sma"])

    if not all(col in data.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in data.columns]
        msg = f"Data must contain required columns. Missing: {missing_cols}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"Running backtest: {strategy.__class__.__name__} on {symbol}")
    logger.info(
        f"Params: {strategy.params}, SL: {stop_loss_pct}, TP: {take_profit_pct}, TSL: {trailing_stop_loss_pct}, Comm: {commission_bps} bps"
    )
    logger.info(
        f"ATR Filter: {apply_atr_filter} (P={atr_filter_period}, M={atr_filter_multiplier}, SMA={atr_filter_sma_period})"
    )
    logger.info(
        f"Seasonality Filter: {apply_seasonality_filter} (Hrs={allowed_trading_hours_utc}, Syms={apply_seasonality_to_symbols})"
    )

    # Parse seasonality params
    parsed_trading_hours = (
        parse_trading_hours(allowed_trading_hours_utc)
        if apply_seasonality_filter
        else None
    )
    seasonality_symbols_list = (
        [s.strip() for s in apply_seasonality_to_symbols.split(",") if s.strip()]
        if apply_seasonality_to_symbols
        else []
    )
    apply_seasonality_to_this_symbol = (
        apply_seasonality_filter
        and parsed_trading_hours
        and (not seasonality_symbols_list or symbol in seasonality_symbols_list)
    )

    # 1. Generate Signals (still needed for entry/exit triggers)
    signals = strategy.generate_signals(data)
    signals = signals.reindex(data.index).fillna(0).astype(int)

    # 2. Initialize Backtest State
    trade_log: List[Dict[str, Any]] = []
    current_pos = 0
    entry_price = 0.0
    entry_time = None
    trade_count = 0
    winning_trades = 0
    balance = initial_balance
    commission_rate = commission_bps / 10000.0
    equity_curve = pd.Series(index=data.index, dtype=float)
    equity_curve.iloc[0] = initial_balance

    stop_loss_level: Optional[float] = None
    take_profit_level: Optional[float] = None
    tsl_peak_price: Optional[float] = None

    # --- Clean SL/TP/TSL Params ---
    def _clean_param(p: Any) -> Optional[float]:
        if isinstance(p, str) and p.lower() == "none":
            return None
        try:
            return float(p) if p is not None else None
        except (ValueError, TypeError):
            return None

    stop_loss_pct_cleaned: Optional[float] = _clean_param(stop_loss_pct)
    take_profit_pct_cleaned: Optional[float] = _clean_param(take_profit_pct)
    trailing_stop_loss_pct_cleaned: Optional[float] = _clean_param(
        trailing_stop_loss_pct
    )

    # Iterate through bars
    for i in range(1, len(data)):
        timestamp = data.index[i]
        prev_timestamp = data.index[i - 1]

        open_price = data["Open"].iloc[i]
        high_price = data["High"].iloc[i]
        low_price = data["Low"].iloc[i]
        close_price = data["Close"].iloc[i]
        prev_close_price = data["Close"].iloc[i - 1]

        if pd.isna(equity_curve.iloc[i]):
            equity_curve.iloc[i] = equity_curve.iloc[i - 1]

        if (
            pd.isna(open_price)
            or pd.isna(high_price)
            or pd.isna(low_price)
            or pd.isna(close_price)
            or pd.isna(prev_close_price)
        ):
            logger.warning(f"NaN price at {timestamp}. Skipping bar.")
            continue

        unrealized_pnl_change = 0
        if current_pos != 0:
            unrealized_pnl_change = (
                (close_price - prev_close_price) * units * current_pos
            )
        equity_curve.iloc[i] += unrealized_pnl_change

        # --- Filtering Logic ---
        trade_allowed_this_bar = True

        # Filter 1: Seasonality
        if apply_seasonality_to_this_symbol:
            start_hour, end_hour = parsed_trading_hours  # type: ignore
            ts_aware = (
                timestamp.tz_convert("UTC")
                if timestamp.tz
                else timestamp.tz_localize("UTC")
            )
            if not (start_hour <= ts_aware.hour < end_hour):
                trade_allowed_this_bar = False
                # logger.debug(f"[{timestamp}] Trade blocked by seasonality filter.")

        # Filter 2: ATR Volatility
        if apply_atr_filter and trade_allowed_this_bar:
            current_atr = data["atr"].iloc[i]
            threshold = (
                data["atr_sma"].iloc[i] * atr_filter_multiplier
                if atr_filter_sma_period > 0 and "atr_sma" in data
                else current_atr * atr_filter_multiplier
            )  # Fallback slightly different if no SMA
            if pd.isna(current_atr) or pd.isna(threshold):
                # logger.debug(f"[{timestamp}] ATR or threshold NaN, cannot apply filter.")
                pass  # Allow trade if ATR is NaN initially?
            elif current_atr < threshold:
                trade_allowed_this_bar = False
                # logger.debug(f"[{timestamp}] Trade blocked by ATR filter (ATR={current_atr:.4f} < Threshold={threshold:.4f}).")

        # --- Check for Exits (SL/TP/TSL) based on intra-bar High/Low ---
        exit_price = None
        exit_reason = None
        if current_pos == 1:  # Long position
            # Update TSL peak price
            if tsl_peak_price is not None:
                tsl_peak_price = max(tsl_peak_price, high_price)
                # Recalculate TSL level based on updated peak
                if isinstance(trailing_stop_loss_pct_cleaned, float):
                    # Add redundant check for tsl_peak_price and explicit cast
                    if tsl_peak_price is not None:
                        tsl_pct_float: float = float(trailing_stop_loss_pct_cleaned)
                        tsl_level = tsl_peak_price * (1 - tsl_pct_float)
                        stop_loss_level = max(
                            stop_loss_level or -np.inf, tsl_level
                        )  # TSL can only move SL up

            # Check exits (prioritize SL > TP)
            if stop_loss_level is not None and low_price <= stop_loss_level:
                exit_price = stop_loss_level  # Exit at SL level
                exit_reason = "Stop Loss"
            elif take_profit_level is not None and high_price >= take_profit_level:
                exit_price = take_profit_level  # Exit at TP level
                exit_reason = "Take Profit"

        elif current_pos == -1:  # Short position
            # Update TSL peak price (using min for shorts)
            if tsl_peak_price is not None:
                tsl_peak_price = min(tsl_peak_price, low_price)
                # Recalculate TSL level based on updated peak
                if isinstance(trailing_stop_loss_pct_cleaned, float):
                    # Add redundant check for tsl_peak_price and explicit cast
                    if tsl_peak_price is not None:
                        tsl_pct_float = float(trailing_stop_loss_pct_cleaned)
                        tsl_level = tsl_peak_price * (1 + tsl_pct_float)
                        stop_loss_level = min(
                            stop_loss_level or np.inf, tsl_level
                        )  # TSL can only move SL down

            # Check exits (prioritize SL > TP)
            if stop_loss_level is not None and high_price >= stop_loss_level:
                exit_price = stop_loss_level  # Exit at SL level
                exit_reason = "Stop Loss"
            elif take_profit_level is not None and low_price <= take_profit_level:
                exit_price = take_profit_level  # Exit at TP level
                exit_reason = "Take Profit"

        # --- Process Exit if Triggered Intra-Bar ---
        if exit_price is not None:
            pnl = (exit_price - entry_price) * units * current_pos
            commission = (
                abs(exit_price * units * commission_rate) * 2
            )  # Entry + Exit commission
            net_pnl = pnl - commission
            balance += net_pnl
            trade_count += 1
            if net_pnl > 0:
                winning_trades += 1

            trade_log.append(
                {
                    "Entry Time": entry_time,
                    "Entry Price": entry_price,
                    "Exit Time": timestamp,
                    "Exit Price": exit_price,
                    "Position": "Long" if current_pos == 1 else "Short",
                    "Units": units,
                    "Gross PnL": pnl,
                    "Commission": commission,
                    "Net PnL": net_pnl,
                    "Exit Reason": exit_reason,
                    "Balance": balance,
                }
            )

            # Update equity curve for the realized PnL
            equity_curve.iloc[i] = balance

            # Reset position state
            current_pos = 0
            entry_price = 0.0
            entry_time = None
            stop_loss_level = None
            take_profit_level = None
            tsl_peak_price = None

        # --- Check for New Signals & Potential Entries (if flat and not filtered) ---
        signal = signals.iloc[i]
        if current_pos == 0 and trade_allowed_this_bar:
            if signal == 1:  # Go Long
                current_pos = 1
                entry_price = close_price  # Enter at close of signal bar
                entry_time = timestamp
                # Set initial SL/TP/TSL levels
                if isinstance(stop_loss_pct_cleaned, float):
                    # Explicitly use the float value
                    stop_loss_level = entry_price * (1 - float(stop_loss_pct_cleaned))
                else:
                    stop_loss_level = None
                if isinstance(take_profit_pct_cleaned, float):
                    # Explicitly use the float value
                    take_profit_level = entry_price * (
                        1 + float(take_profit_pct_cleaned)
                    )
                else:
                    take_profit_level = None
                tsl_peak_price = (
                    entry_price
                    if isinstance(trailing_stop_loss_pct_cleaned, float)
                    else None
                )

            elif signal == -1:  # Go Short
                current_pos = -1
                entry_price = close_price  # Enter at close of signal bar
                entry_time = timestamp
                # Set initial SL/TP/TSL levels
                if isinstance(stop_loss_pct_cleaned, float):
                    stop_loss_level = entry_price * (1 + float(stop_loss_pct_cleaned))
                else:
                    stop_loss_level = None
                if isinstance(take_profit_pct_cleaned, float):
                    # Add assert for type checker
                    assert isinstance(take_profit_pct_cleaned, float)
                    # Add assert for entry_price as well
                    assert isinstance(entry_price, float)  # type: ignore
                    # Ignoring persistent mypy errors here on the calculation line
                    take_profit_level = entry_price * (1 - take_profit_pct_cleaned)  # type: ignore
                else:
                    take_profit_level = None
                tsl_peak_price = (
                    entry_price
                    if isinstance(trailing_stop_loss_pct_cleaned, float)
                    else None
                )

        # --- Check for Exit Signal (if in position and not exited by SL/TP/TSL) ---
        elif current_pos != 0:
            exit_signal = False
            if current_pos == 1 and signal == -1:
                exit_signal = True
            if current_pos == -1 and signal == 1:
                exit_signal = True

            if exit_signal:
                exit_price = close_price  # Exit at close of signal bar
                pnl = (exit_price - entry_price) * units * current_pos
                commission = abs(exit_price * units * commission_rate) * 2
                net_pnl = pnl - commission
                balance += net_pnl
                trade_count += 1
                if net_pnl > 0:
                    winning_trades += 1

                trade_log.append(
                    {
                        "Entry Time": entry_time,
                        "Entry Price": entry_price,
                        "Exit Time": timestamp,
                        "Exit Price": exit_price,
                        "Position": "Long" if current_pos == 1 else "Short",
                        "Units": units,
                        "Gross PnL": pnl,
                        "Commission": commission,
                        "Net PnL": net_pnl,
                        "Exit Reason": "Signal",
                        "Balance": balance,
                    }
                )
                equity_curve.iloc[i] = balance
                current_pos = 0
                entry_price = 0.0
                entry_time = None
                stop_loss_level = None
                take_profit_level = None
                tsl_peak_price = None

    # --- Final Calculations & Summary ---
    equity_curve.ffill(inplace=True)  # Fill any NaNs from skipped bars
    total_pnl = balance - initial_balance
    win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0.0
    daily_returns = equity_curve.pct_change().dropna()
    sharpe = calculate_sharpe_ratio(daily_returns)
    max_dd = calculate_max_drawdown(equity_curve)

    performance_df = pd.DataFrame(trade_log)
    profit_factor = calculate_profit_factor(performance_df)

    logger.info(
        f"Backtest complete. Final Balance: {balance:.2f}, Total PnL: {total_pnl:.2f}, Trades: {trade_count}, Win Rate: {win_rate:.2f}%"
    )

    return {
        "cumulative_profit": total_pnl,
        "final_balance": balance,
        "total_trades": trade_count,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "performance_summary": performance_df,
        "equity_curve": equity_curve,
    }
