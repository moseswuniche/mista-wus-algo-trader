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
    # Ensure standard lowercase column names
    df.rename(
        columns={
            "Timestamp": "timestamp",  # Ensure timestamp column if it exists
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
    # Verify required columns exist after renaming
    required_cols = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"Dataframe missing required columns after renaming: {missing}")
        return None

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

    # --- Vectorized Iteration (Itertuples) --- #
    logger.debug("Starting backtest loop...")
    for i, row in enumerate(df.itertuples()):
        # --- Pre-calculation for current step --- #
        current_time = row.Index
        current_price = row.close
        signal = getattr(row, "signal", 0)
        high_price = row.high
        low_price = row.low

        # --- Update Equity Curve --- #
        # Use previous balance if not the first row
        prev_balance = equity_curve.iloc[i - 1] if i > 0 else initial_balance
        # Calculate current equity based on open position PnL
        if current_pos != 0:
            pnl = current_pos * (current_price - entry_price) * units
            current_equity = prev_balance + pnl
        else:
            current_equity = prev_balance
        equity_curve.iloc[i] = current_equity

        # --- Apply Filters (Skip signals if filters active) --- #
        filtered_out = False
        # ATR Filter Check
        if apply_atr_filter and hasattr(row, "atr") and row.atr is not None:
            atr_threshold = row.atr * atr_filter_multiplier
            if atr_filter_sma_period > 0 and hasattr(row, "atr_sma") and row.atr_sma is not None:
                # If SMA is enabled, allow trade only if price is above/below ATR threshold relative to SMA
                # This part might need refinement based on the exact ATR filter logic desired
                # Example: require price to be significantly volatile compared to recent average volatility
                # Simplified version: Check if ATR itself is above the threshold (basic volatility check)
                if row.atr < (atr_threshold / atr_filter_multiplier): # Check original ATR value for threshold
                    filtered_out = True
                    # logger.debug(f"{current_time}: Filtered by ATR (value {row.atr:.4f} < threshold {atr_threshold:.4f}) using SMA baseline") # COMMENTED OUT
            elif row.atr < atr_threshold:
                    filtered_out = True
                    # logger.debug(f"{current_time}: Filtered by ATR (value {row.atr:.4f} < threshold {atr_threshold:.4f})") # COMMENTED OUT

        # Seasonality Filter Check
        if (
            apply_seasonality_to_this_symbol
            and not filtered_out
            and parsed_trading_hours
        ):
            current_hour = current_time.hour
            start_hour, end_hour = parsed_trading_hours
            if not (start_hour <= current_hour < end_hour):
                filtered_out = True
                # logger.debug(f"{current_time}: Filtered by Seasonality (hour {current_hour} outside {start_hour}-{end_hour})") # COMMENTED OUT
                # If filtered by seasonality, force exit any open position
                signal = -current_pos if current_pos != 0 else 0 # Force exit

        # --- Check for Stop Loss / Take Profit / Trailing Stop --- #
        exit_reason = None
        exit_price = current_price # Default exit price if triggered by signal change

        if current_pos == 1: # Long position checks
            # Trailing Stop Loss Update & Check
            if trailing_stop_loss_pct_cleaned is not None:
                if tsl_peak_price is None or high_price > tsl_peak_price:
                    tsl_peak_price = high_price
                potential_tsl_level = tsl_peak_price * (1 - trailing_stop_loss_pct_cleaned)
                if stop_loss_level is None or potential_tsl_level > stop_loss_level:
                    # if stop_loss_level != potential_tsl_level: # Check prevents logging spam if level unchanged
                    #      logger.debug(f"{current_time}: Updated TSL level (Long) to {potential_tsl_level:.4f} from {stop_loss_level}") # COMMENTED OUT
                    stop_loss_level = potential_tsl_level

            # Stop Loss Check
            if stop_loss_level is not None and low_price <= stop_loss_level:
                exit_reason = "Stop Loss"
                exit_price = stop_loss_level # Exit at stop level
                signal = -1 # Force exit signal
            # Take Profit Check
            elif take_profit_level is not None and high_price >= take_profit_level:
                exit_reason = "Take Profit"
                exit_price = take_profit_level # Exit at profit level
                signal = -1 # Force exit signal

        elif current_pos == -1: # Short position checks
            # Trailing Stop Loss Update & Check
            if trailing_stop_loss_pct_cleaned is not None:
                if tsl_peak_price is None or low_price < tsl_peak_price:
                    tsl_peak_price = low_price
                potential_tsl_level = tsl_peak_price * (1 + trailing_stop_loss_pct_cleaned)
                if stop_loss_level is None or potential_tsl_level < stop_loss_level:
                    # if stop_loss_level != potential_tsl_level: # Check prevents logging spam if level unchanged
                    #     logger.debug(f"{current_time}: Updated TSL level (Short) to {potential_tsl_level:.4f} from {stop_loss_level}") # COMMENTED OUT
                    stop_loss_level = potential_tsl_level

            # Stop Loss Check
            if stop_loss_level is not None and high_price >= stop_loss_level:
                exit_reason = "Stop Loss"
                exit_price = stop_loss_level # Exit at stop level
                signal = 1 # Force exit signal
            # Take Profit Check
            elif take_profit_level is not None and low_price <= take_profit_level:
                exit_reason = "Take Profit"
                exit_price = take_profit_level # Exit at profit level
                signal = 1 # Force exit signal

        # --- Position Management based on Signal --- #
        if filtered_out and signal != -current_pos: # If filtered out, ignore new entry signals
             if signal != 0 and current_pos == 0 :
                 # logger.debug(f"{current_time}: Entry signal {signal} ignored due to active filter.") # COMMENTED OUT
                 pass # Explicitly do nothing
             signal = 0 # Ignore entry signal if filtered

        if current_pos == 0:
            if signal == 1:
                # Enter Long
                current_pos = 1
                entry_price = current_price
                entry_time = current_time
                trade_count += 1
                commission_cost = entry_price * units * commission
                balance -= commission_cost
                stop_loss_level = entry_price * (1 - stop_loss_pct_cleaned) if stop_loss_pct_cleaned is not None else None
                take_profit_level = entry_price * (1 + take_profit_pct_cleaned) if take_profit_pct_cleaned is not None else None
                tsl_peak_price = entry_price # Initialize TSL peak
                # logger.debug(f"{current_time}: ENTER LONG @ {entry_price:.4f}, Units: {units}, Commission: {commission_cost:.4f}, Balance: {balance:.2f}, SL: {stop_loss_level}, TP: {take_profit_level}") # COMMENTED OUT
            elif signal == -1:
                # Enter Short
                current_pos = -1
                entry_price = current_price
                entry_time = current_time
                trade_count += 1
                commission_cost = entry_price * units * commission
                balance -= commission_cost
                stop_loss_level = entry_price * (1 + stop_loss_pct_cleaned) if stop_loss_pct_cleaned is not None else None
                take_profit_level = entry_price * (1 - take_profit_pct_cleaned) if take_profit_pct_cleaned is not None else None
                tsl_peak_price = entry_price # Initialize TSL peak
                # logger.debug(f"{current_time}: ENTER SHORT @ {entry_price:.4f}, Units: {units}, Commission: {commission_cost:.4f}, Balance: {balance:.2f}, SL: {stop_loss_level}, TP: {take_profit_level}") # COMMENTED OUT

        elif current_pos == 1 and signal == -1:
            # Exit Long
            pnl = (exit_price - entry_price) * units
            commission_cost = exit_price * units * commission
            net_pnl = pnl - commission_cost
            balance += net_pnl
            if net_pnl > 0: winning_trades += 1
            trade_log.append(
                {
                    "Entry Time": entry_time,
                    "Exit Time": current_time,
                    "Direction": "Long",
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Profit/Loss": net_pnl,
                    "Commission": commission_cost,
                    "Exit Reason": exit_reason or "Signal",
                }
            )
            # logger.debug(f"{current_time}: EXIT LONG @ {exit_price:.4f} from {entry_price:.4f}, Reason: {exit_reason or 'Signal'}, PnL: {net_pnl:.2f}, Commission: {commission_cost:.4f}, Balance: {balance:.2f}") # COMMENTED OUT
            current_pos = 0
            entry_price = 0.0
            entry_time = None
            stop_loss_level = None
            take_profit_level = None
            tsl_peak_price = None
            # Check if we need to enter short immediately
            if not filtered_out and getattr(row, "signal", 0) == -1: # Use original signal if not forced exit
                 # Enter Short immediately after exit
                 current_pos = -1
                 entry_price = current_price # Use current row close for immediate entry
                 entry_time = current_time
                 trade_count += 1
                 commission_cost_entry = entry_price * units * commission
                 balance -= commission_cost_entry
                 stop_loss_level = entry_price * (1 + stop_loss_pct_cleaned) if stop_loss_pct_cleaned is not None else None
                 take_profit_level = entry_price * (1 - take_profit_pct_cleaned) if take_profit_pct_cleaned is not None else None
                 tsl_peak_price = entry_price
                 # logger.debug(f"{current_time}: ENTER SHORT (Immediate) @ {entry_price:.4f}, Units: {units}, Commission: {commission_cost_entry:.4f}, Balance: {balance:.2f}, SL: {stop_loss_level}, TP: {take_profit_level}") # COMMENTED OUT


        elif current_pos == -1 and signal == 1:
            # Exit Short
            pnl = (entry_price - exit_price) * units # Reversed for short
            commission_cost = exit_price * units * commission
            net_pnl = pnl - commission_cost
            balance += net_pnl
            if net_pnl > 0: winning_trades += 1
            trade_log.append(
                {
                    "Entry Time": entry_time,
                    "Exit Time": current_time,
                    "Direction": "Short",
                    "Entry Price": entry_price,
                    "Exit Price": exit_price,
                    "Profit/Loss": net_pnl,
                    "Commission": commission_cost,
                    "Exit Reason": exit_reason or "Signal",
                }
            )
            # logger.debug(f"{current_time}: EXIT SHORT @ {exit_price:.4f} from {entry_price:.4f}, Reason: {exit_reason or 'Signal'}, PnL: {net_pnl:.2f}, Commission: {commission_cost:.4f}, Balance: {balance:.2f}") # COMMENTED OUT
            current_pos = 0
            entry_price = 0.0
            entry_time = None
            stop_loss_level = None
            take_profit_level = None
            tsl_peak_price = None
             # Check if we need to enter long immediately
            if not filtered_out and getattr(row, "signal", 0) == 1: # Use original signal if not forced exit
                 # Enter Long immediately after exit
                 current_pos = 1
                 entry_price = current_price # Use current row close for immediate entry
                 entry_time = current_time
                 trade_count += 1
                 commission_cost_entry = entry_price * units * commission
                 balance -= commission_cost_entry
                 stop_loss_level = entry_price * (1 - stop_loss_pct_cleaned) if stop_loss_pct_cleaned is not None else None
                 take_profit_level = entry_price * (1 + take_profit_pct_cleaned) if take_profit_pct_cleaned is not None else None
                 tsl_peak_price = entry_price
                 # logger.debug(f"{current_time}: ENTER LONG (Immediate) @ {entry_price:.4f}, Units: {units}, Commission: {commission_cost_entry:.4f}, Balance: {balance:.2f}, SL: {stop_loss_level}, TP: {take_profit_level}") # COMMENTED OUT

        # Log equity at each step (or less frequently if needed)
        # logger.debug(f"{current_time}: Equity: {equity_curve.iloc[i]:.2f}, Balance: {balance:.2f}, Position: {current_pos}") # COMMENTED OUT

    # --- End Backtest Loop --- #
    logger.debug("Backtest loop finished.")

    # --- Final Equity Calculation --- #
    # If still in position at the end, calculate final equity based on last price
    if current_pos != 0:
        last_price = df["close"].iloc[-1]
        pnl = current_pos * (last_price - entry_price) * units
        # No exit commission applied here, assuming position is marked-to-market
        final_equity = balance + pnl  # Use final balance before closing PnL
        equity_curve.iloc[-1] = final_equity  # Update last equity point
        logger.debug(
            f"End of data: Marked-to-market final equity: {final_equity:.2f} (Position {current_pos})"
        )
    else:
        final_equity = balance  # Final equity is the final balance if flat
        equity_curve.iloc[-1] = final_equity
        logger.debug(f"End of data: Final equity (flat): {final_equity:.2f}")

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
