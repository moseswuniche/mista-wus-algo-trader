# mypy: disable-error-code=operator
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, cast, Tuple, Union
import logging
from datetime import time  # For Seasonality
import inspect

# --- Add type ignore for the whole file if necessary, or specific function ---
# type: ignore[operator]

from .strategies.base_strategy import Strategy
from .technical_indicators import (
    calculate_atr,
    calculate_sma,
    calculate_ema,
)  # Import ATR function
from .strategies import (
    LongShortStrategy,
    MovingAverageCrossoverStrategy,
    RsiMeanReversionStrategy,
    BollingerBandReversionStrategy,
)  # Import for type checking

# Import the configuration model
from .config_models import BacktestRunConfig, ValidationError

# Get logger instance
logger = logging.getLogger(__name__)

# Map strategy short names (used in args) to classes
# This map is needed here to instantiate the strategy from config
# Consider moving this map to a central location (e.g., config_models or a dedicated strategies module)
STRATEGY_MAP = {
    "LongShort": LongShortStrategy,
    "MACross": MovingAverageCrossoverStrategy,
    "RSIReversion": RsiMeanReversionStrategy,
    "BBReversion": BollingerBandReversionStrategy,
}


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


# --- Main Backtest Function --- UPDATED for Option A ---
def run_backtest(
    data: pd.DataFrame,
    config: BacktestRunConfig,
) -> Optional[Dict[str, Any]]:
    """Runs a vectorized backtest for a given strategy configuration.

    Args:
        data: DataFrame with historical OHLCV data, index must be DatetimeIndex.
        config: BacktestRunConfig object containing all run parameters.

    Returns:
        Dictionary containing performance metrics, or None if backtest fails.
    """
    logger.debug(
        f"--- Starting Backtest Run for: {config.symbol} / {config.strategy_short_name} ---"
    )
    logger.debug(f"Backtest Config: {config.model_dump()}")  # Log the full config

    # --- Input Validation & Data Preparation --- #
    if data.empty:
        logger.error("Input data is empty. Cannot run backtest.")
        return None
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.error("Data index must be a DatetimeIndex.")
        return None

    # --- Instantiate Strategy from Config ---
    strategy_class = STRATEGY_MAP.get(config.strategy_short_name)
    if not strategy_class:
        logger.error(f"Strategy short name '{config.strategy_short_name}' not found.")
        return None
    try:
        # Make sure params don't accidentally contain non-init args
        sig = inspect.signature(strategy_class.__init__)  # type: ignore[misc]
        valid_init_params = {p for p in sig.parameters if p != "self"}
        strategy_init_args = {
            k: v for k, v in config.strategy_params.items() if k in valid_init_params
        }
        strategy = strategy_class(**strategy_init_args)
        logger.debug(
            f"Instantiated strategy {config.strategy_short_name} with params: {strategy_init_args}"
        )
    except Exception as e:
        logger.error(
            f"Error instantiating strategy '{config.strategy_short_name}' with params {config.strategy_params}: {e}",
            exc_info=True,
        )
        return None
    # --- End Instantiate Strategy ---

    # --- Unpack Config Values ---
    symbol = config.symbol
    initial_balance = config.initial_balance
    commission_bps = config.commission_bps
    units = config.units
    stop_loss_pct = config.stop_loss_pct
    take_profit_pct = config.take_profit_pct
    trailing_stop_loss_pct = config.trailing_stop_loss_pct
    apply_atr_filter = config.apply_atr_filter
    atr_filter_period = config.atr_filter_period
    atr_filter_multiplier = config.atr_filter_multiplier
    atr_filter_sma_period = config.atr_filter_sma_period
    apply_seasonality_filter = config.apply_seasonality_filter
    allowed_trading_hours_utc = config.allowed_trading_hours_utc
    apply_seasonality_to_symbols = config.apply_seasonality_to_symbols
    # --- End Unpack Config ---

    # --- Existing Data Prep & Indicator Calculation --- #
    df = data.copy()
    commission = commission_bps / 10000.0  # Convert basis points to decimal

    # --- Prepare Filter-Related Data --- #
    if apply_atr_filter:
        df["atr"] = calculate_atr(df, period=atr_filter_period)
        if atr_filter_sma_period > 0:
            df["atr_sma"] = calculate_sma(df["atr"], period=atr_filter_sma_period)
        logger.debug("Calculated ATR for filter.")

    parsed_trading_hours = parse_trading_hours(allowed_trading_hours_utc)
    seasonality_symbols_list = (
        [
            s.strip().upper()
            for s in apply_seasonality_to_symbols.split(",")
            if s.strip()
        ]
        if apply_seasonality_to_symbols
        else []
    )
    apply_seasonality_to_this_symbol = (
        apply_seasonality_filter
        and parsed_trading_hours
        and (not seasonality_symbols_list or symbol in seasonality_symbols_list)
    )
    if apply_seasonality_filter and not parsed_trading_hours:
        logger.warning(
            f"Seasonality filter enabled but invalid allowed_trading_hours_utc: '{allowed_trading_hours_utc}'. Filter inactive."
        )
        apply_seasonality_to_this_symbol = False  # Disable if hours invalid
    # --- End Filter Data Prep ---

    # --- Generate Signals --- #
    try:
        df = strategy.generate_signals(df)
        if "signal" not in df.columns:
            logger.error("Strategy did not generate 'signal' column.")
            return None
        logger.debug("Generated strategy signals.")
    except Exception as e:
        logger.error(
            f"Error generating signals for strategy {strategy.__class__.__name__}: {e}",
            exc_info=True,
        )
        return None
    # --- End Generate Signals ---

    # --- Initialize Backtest State Columns --- #
    trade_log: List[Dict[str, Any]] = []
    current_pos = 0
    entry_price = 0.0
    entry_time = None
    trade_count = 0
    winning_trades = 0
    balance = initial_balance
    equity_curve = pd.Series(index=df.index, dtype=float)
    equity_curve.iloc[0] = initial_balance

    stop_loss_level: Optional[float] = None
    take_profit_level: Optional[float] = None
    tsl_peak_price: Optional[float] = None

    # --- Clean SL/TP/TSL Params (Use EFFECTIVE values) ---
    def _clean_param(p: Any) -> Optional[float]:
        if isinstance(p, str) and p.lower() == "none":
            return None
        try:
            return float(p) if p is not None else None
        except (ValueError, TypeError):
            return None

    # Use the determined effective values
    stop_loss_pct_cleaned: Optional[float] = _clean_param(stop_loss_pct)
    take_profit_pct_cleaned: Optional[float] = _clean_param(take_profit_pct)
    trailing_stop_loss_pct_cleaned: Optional[float] = _clean_param(
        trailing_stop_loss_pct
    )

    # Iterate through bars (use df length)
    for i in range(1, len(df)):
        timestamp = df.index[i]
        prev_timestamp = df.index[i - 1]

        # Use df for prices
        open_price = df["Open"].iloc[i]
        high_price = df["High"].iloc[i]
        low_price = df["Low"].iloc[i]
        close_price = df["Close"].iloc[i]
        prev_close_price = df["Close"].iloc[i - 1]

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

        # --- Filtering Logic (Use EFFECTIVE settings) ---
        trade_allowed_this_bar = True

        # Filter 1: Seasonality (Uses effective settings)
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

        # Filter 2: ATR Volatility (Uses effective settings and RECALCULATED columns)
        if apply_atr_filter and trade_allowed_this_bar:
            # Use the recalculated 'atr' column
            current_atr = df["atr"].iloc[i]
            threshold = 0.0  # Initialize

            if atr_filter_sma_period > 0:
                # Use the recalculated 'atr_sma' column for the baseline
                atr_sma_effective_val = df["atr_sma"].iloc[i]
                if not pd.isna(atr_sma_effective_val):
                    threshold = atr_sma_effective_val * atr_filter_multiplier
                else:
                    # Fallback if SMA is NaN (e.g., during initial window)
                    # Use current ATR * multiplier as threshold only if current ATR is not NaN
                    if not pd.isna(current_atr):
                        threshold = current_atr * atr_filter_multiplier
                    else:
                        threshold = np.nan  # Keep threshold as NaN if both are NaN

            else:
                # If no SMA period, use current ATR * multiplier if current ATR is not NaN
                if not pd.isna(current_atr):
                    threshold = current_atr * atr_filter_multiplier
                else:
                    threshold = np.nan  # Keep threshold as NaN if current ATR is NaN

            # Check if trade is allowed
            if pd.isna(current_atr) or pd.isna(threshold):
                # If ATR or threshold is NaN (e.g., at the start), decide whether to allow trade.
                # Current behavior: Allow trade (pass). Consider blocking if needed.
                pass
                # logger.debug(f"[{timestamp}] ATR or threshold NaN, filter not applied.")
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
                abs(exit_price * units * commission) * 2
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
        signal = df["signal"].iloc[i]
        if current_pos == 0 and trade_allowed_this_bar:
            if signal == 1:  # Go Long
                current_pos = 1
                entry_price = close_price  # Enter at close of signal bar
                entry_time = timestamp
                # Set initial SL/TP/TSL levels based on _cleaned effective values
                if isinstance(stop_loss_pct_cleaned, float):
                    # Explicitly use the float value
                    stop_loss_level = entry_price * (1 - float(stop_loss_pct_cleaned))
                else:
                    stop_loss_level = None
                if isinstance(take_profit_pct_cleaned, float):
                    # Add assert for type checker back
                    assert isinstance(take_profit_pct_cleaned, float)
                    # Add assert for entry_price as well
                    assert isinstance(entry_price, float)
                    # Directly use the variable known to be float
                    take_profit_level = entry_price * (1 - take_profit_pct_cleaned)  # type: ignore[operator]
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
                # Set initial SL/TP/TSL levels based on _cleaned effective values
                if isinstance(stop_loss_pct_cleaned, float):
                    stop_loss_level = entry_price * (1 + float(stop_loss_pct_cleaned))
                else:
                    stop_loss_level = None
                if isinstance(take_profit_pct_cleaned, float):
                    # Add assert for type checker back
                    assert isinstance(take_profit_pct_cleaned, float)
                    # Add assert for entry_price as well
                    assert isinstance(entry_price, float)
                    # Directly use the variable known to be float
                    take_profit_level = entry_price * (1 - take_profit_pct_cleaned)  # type: ignore[operator]
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
                commission = abs(exit_price * units * commission) * 2
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

    # Add Rolling Sharpe Calculation
    SHARPE_ROLLING_WINDOW = 90  # Lookback window in days for rolling Sharpe
    RISK_FREE_RATE = 0.0
    ANNUALIZATION_FACTOR = 365  # Assuming daily data

    # Calculate daily returns from equity curve
    daily_returns = equity_curve.pct_change()

    # Calculate rolling mean of *excess* returns
    excess_daily_returns = daily_returns - (RISK_FREE_RATE / ANNUALIZATION_FACTOR)
    rolling_excess_mean = excess_daily_returns.rolling(
        window=SHARPE_ROLLING_WINDOW
    ).mean()

    # Calculate rolling mean and std dev
    rolling_std = daily_returns.rolling(window=SHARPE_ROLLING_WINDOW).std()

    # Calculate only where std is positive to avoid division issues
    rolling_sharpe_non_annualized = pd.Series(np.nan, index=equity_curve.index)
    valid_mask = rolling_std > 0
    # Ensure alignment by using .loc with the mask for all series involved
    rolling_sharpe_non_annualized.loc[valid_mask] = (
        rolling_excess_mean.loc[valid_mask] / rolling_std.loc[valid_mask]
    )

    # Fill NaNs (from window or zero std) with 0 and annualize
    rolling_sharpe_annualized = (
        rolling_sharpe_non_annualized * np.sqrt(ANNUALIZATION_FACTOR)
    ).fillna(0.0)

    total_pnl = balance - initial_balance
    win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0.0
    # Use the already calculated daily_returns, dropping NaNs for overall calculation
    daily_returns_dropna = daily_returns.dropna()
    sharpe = calculate_sharpe_ratio(daily_returns_dropna)
    # Now pass the original, unmodified equity_curve Series
    max_dd = calculate_max_drawdown(equity_curve)

    performance_df = pd.DataFrame(trade_log)
    profit_factor = calculate_profit_factor(performance_df)

    logger.info(
        f"Backtest complete. Final Balance: {balance:.2f}, Total PnL: {total_pnl:.2f}, Trades: {trade_count}, Win Rate: {win_rate:.2f}%"
    )

    # Create a combined DataFrame for equity and rolling Sharpe
    equity_curve_df = pd.DataFrame(
        {"equity": equity_curve, "rolling_sharpe": rolling_sharpe_annualized}
    )

    results = {
        "cumulative_profit": total_pnl,
        "final_balance": balance,
        "total_trades": trade_count,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "performance_summary": performance_df,
        "equity_curve": equity_curve_df,
    }

    logger.debug(
        f"--- Backtest Run Finished for: {config.symbol} / {config.strategy_short_name} ---"
    )
    return results
