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


# --- Main Backtest Function --- UPDATED for Option A ---
def run_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    symbol: str,
    units: float,
    initial_balance: float = 10000.0,
    commission_bps: float = 0.0,
    # --- Global Defaults / Command-Line Args ---
    # These are kept for fallback and context
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    trailing_stop_loss_pct: Optional[float] = None,
    apply_atr_filter: bool = False,
    atr_filter_period: int = 14,
    atr_filter_multiplier: float = 1.5,
    atr_filter_sma_period: int = 100,
    apply_seasonality_filter: bool = False,
    allowed_trading_hours_utc: Optional[str] = None,
    apply_seasonality_to_symbols: Optional[str] = None,
    # --- Grid-Specific Parameters (Passed from Worker) ---
    # These will override the global defaults if provided (not None)
    sl_from_grid: Optional[float] = None,
    tp_from_grid: Optional[float] = None,
    tsl_from_grid: Optional[float] = None,
    apply_atr_from_grid: Optional[bool] = None,
    atr_period_from_grid: Optional[int] = None,
    atr_multiplier_from_grid: Optional[float] = None,
    atr_sma_from_grid: Optional[int] = None,
    apply_seasonality_from_grid: Optional[bool] = None,
    trading_hours_from_grid: Optional[str] = None,
    seasonality_symbols_from_grid: Optional[str] = None,  # Added for completeness
) -> Dict[str, Any]:
    """
    Runs a bar-by-bar backtest simulating intra-bar SL/TP/TSL checks and applying filters.
    Prioritizes '_from_grid' parameters if they are provided (not None), otherwise falls back
    to the global default parameters (e.g., from command line).

    Args:
        data: DataFrame with historical OHLCV data, indexed by Date.
              Must include 'Open', 'High', 'Low', 'Close'.
              Must also contain 'atr' and 'atr_sma' columns if apply_atr_filter is True.
        strategy: An instantiated strategy object.
        symbol: The symbol being backtested (e.g., 'BTCUSDT').
        units: The fixed amount of the asset to trade per signal.
        initial_balance: Starting balance for calculating returns.
        commission_bps: Commission fee in basis points (e.g., 10 for 0.1%).
        stop_loss_pct: GLOBAL default stop loss percentage.
        take_profit_pct: GLOBAL default take profit percentage.
        trailing_stop_loss_pct: GLOBAL default trailing stop loss percentage.
        apply_atr_filter: GLOBAL default setting for ATR filter.
        atr_filter_period: GLOBAL default period for ATR calculation.
        atr_filter_multiplier: GLOBAL default multiplier for ATR volatility threshold.
        atr_filter_sma_period: GLOBAL default SMA period for ATR threshold baseline.
        apply_seasonality_filter: GLOBAL default setting for seasonality filter.
        allowed_trading_hours_utc: GLOBAL default string for allowed UTC hours.
        apply_seasonality_to_symbols: GLOBAL default string for symbols for seasonality.
        sl_from_grid: Stop loss percentage from the optimization grid for this run.
        tp_from_grid: Take profit percentage from the grid for this run.
        tsl_from_grid: Trailing stop loss percentage from the grid for this run.
        apply_atr_from_grid: ATR filter setting from the grid for this run.
        atr_period_from_grid: ATR period from the grid for this run.
        atr_multiplier_from_grid: ATR multiplier from the grid for this run.
        atr_sma_from_grid: ATR SMA period from the grid for this run.
        apply_seasonality_from_grid: Seasonality filter setting from the grid for this run.
        trading_hours_from_grid: Trading hours string from the grid for this run.
        seasonality_symbols_from_grid: Seasonality symbols string from the grid for this run.

    Returns:
        A dictionary containing backtest results.
    """
    # --- Determine Effective Parameters (Grid > Global Default) ---
    effective_sl = sl_from_grid if sl_from_grid is not None else stop_loss_pct
    effective_tp = tp_from_grid if tp_from_grid is not None else take_profit_pct
    effective_tsl = (
        tsl_from_grid if tsl_from_grid is not None else trailing_stop_loss_pct
    )

    effective_apply_atr = (
        apply_atr_from_grid if apply_atr_from_grid is not None else apply_atr_filter
    )
    effective_atr_period = (
        atr_period_from_grid if atr_period_from_grid is not None else atr_filter_period
    )
    effective_atr_multiplier = (
        atr_multiplier_from_grid
        if atr_multiplier_from_grid is not None
        else atr_filter_multiplier
    )
    effective_atr_sma_period = (
        atr_sma_from_grid if atr_sma_from_grid is not None else atr_filter_sma_period
    )

    effective_apply_seasonality = (
        apply_seasonality_from_grid
        if apply_seasonality_from_grid is not None
        else apply_seasonality_filter
    )
    effective_trading_hours = (
        trading_hours_from_grid
        if trading_hours_from_grid is not None
        else allowed_trading_hours_utc
    )
    effective_seasonality_symbols = (
        seasonality_symbols_from_grid
        if seasonality_symbols_from_grid is not None
        else apply_seasonality_to_symbols
    )
    # --- End Effective Parameter Determination ---

    required_cols = ["Open", "High", "Low", "Close"]
    # --- Recalculate ATR/SMA if needed (Option 1) ---
    # Make a copy to avoid modifying the original data potentially shared across processes
    local_data = data.copy()
    if effective_apply_atr:
        # Ensure the technical_indicators module and calculate_atr function are imported
        from .technical_indicators import calculate_atr # Ensure import is here

        logger.info(
            f"Recalculating ATR with period {effective_atr_period} for this run."
        )
        # Recalculate ATR using the effective period
        local_data["atr_effective"] = calculate_atr(
            local_data, period=effective_atr_period
        )
        # Ensure 'atr_effective' is now the required column, not 'atr'
        if "atr" in required_cols: required_cols.remove("atr")
        required_cols.append("atr_effective")


        if effective_atr_sma_period > 0:
            logger.info(
                f"Recalculating ATR SMA with period {effective_atr_sma_period} for this run."
            )
            # Recalculate ATR SMA using the effective SMA period on the *newly calculated* ATR
            local_data["atr_sma_effective"] = (
                local_data["atr_effective"]
                .rolling(window=effective_atr_sma_period, min_periods=effective_atr_sma_period) # Add min_periods
                .mean()
            )
             # Ensure 'atr_sma_effective' is now the required column, not 'atr_sma'
            if "atr_sma" in required_cols: required_cols.remove("atr_sma")
            required_cols.append("atr_sma_effective")

        # Remove the potentially confusing warning about pre-calculated columns
        # logger.warning(...) - removed


    # Check data requirements based on EFFECTIVE settings
    # if effective_apply_atr: # This block is now handled above
    #     logger.warning(
    #         f"Using pre-calculated 'atr' column. Ensure it corresponds to effective ATR period: {effective_atr_period}"
    #     )
    #     required_cols.extend(["atr"])
    #     if effective_atr_sma_period > 0:
    #         logger.warning(
    #             f"Using pre-calculated 'atr_sma' column. Ensure it corresponds to effective ATR SMA period: {effective_atr_sma_period}"
    #         )
    #         required_cols.extend(["atr_sma"])

    # Check against local_data which now potentially has recalculated indicators
    if not all(col in local_data.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in local_data.columns]
        msg = f"Data must contain required columns based on effective settings. Missing: {missing_cols}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"Running backtest: {strategy.__class__.__name__} on {symbol}")
    # Log the EFFECTIVE SL/TP/TSL values being used for this run
    logger.info(
        f"Effective Params -> SL: {effective_sl}, TP: {effective_tp}, TSL: {effective_tsl}, Comm: {commission_bps} bps"
    )
    # Log the EFFECTIVE Filter settings
    logger.info(
        f"Effective ATR Filter: {effective_apply_atr} (P={effective_atr_period}, M={effective_atr_multiplier}, SMA={effective_atr_sma_period})"
    )
    logger.info(
        f"Effective Seasonality Filter: {effective_apply_seasonality} (Hrs={effective_trading_hours}, Syms={effective_seasonality_symbols})"
    )

    # Parse seasonality params based on EFFECTIVE settings
    parsed_trading_hours = (
        parse_trading_hours(effective_trading_hours)
        if effective_apply_seasonality
        else None
    )
    seasonality_symbols_list = (
        [s.strip() for s in effective_seasonality_symbols.split(",") if s.strip()]
        if effective_seasonality_symbols
        else []
    )
    # Determine if seasonality applies to THIS symbol based on EFFECTIVE settings
    apply_seasonality_to_this_symbol = (
        effective_apply_seasonality
        and parsed_trading_hours
        and (not seasonality_symbols_list or symbol in seasonality_symbols_list)
    )

    # 1. Generate Signals (still needed for entry/exit triggers)
    # Use local_data which might have recalculated indicators if needed by strategy
    signals = strategy.generate_signals(local_data)
    signals = signals.reindex(local_data.index).fillna(0).astype(int)

    # 2. Initialize Backtest State
    trade_log: List[Dict[str, Any]] = []
    current_pos = 0
    entry_price = 0.0
    entry_time = None
    trade_count = 0
    winning_trades = 0
    balance = initial_balance
    commission_rate = commission_bps / 10000.0
    # Initialize using local_data index
    equity_curve = pd.Series(index=local_data.index, dtype=float)
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
    stop_loss_pct_cleaned: Optional[float] = _clean_param(effective_sl)
    take_profit_pct_cleaned: Optional[float] = _clean_param(effective_tp)
    trailing_stop_loss_pct_cleaned: Optional[float] = _clean_param(effective_tsl)

    # Iterate through bars (use local_data length)
    for i in range(1, len(local_data)):
        timestamp = local_data.index[i]
        prev_timestamp = local_data.index[i - 1]

        # Use local_data for prices
        open_price = local_data["Open"].iloc[i]
        high_price = local_data["High"].iloc[i]
        low_price = local_data["Low"].iloc[i]
        close_price = local_data["Close"].iloc[i]
        prev_close_price = local_data["Close"].iloc[i - 1]

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
        if effective_apply_atr and trade_allowed_this_bar:
            # Use the recalculated 'atr_effective' column
            current_atr = local_data["atr_effective"].iloc[i]
            threshold = 0.0 # Initialize

            if effective_atr_sma_period > 0:
                # Use the recalculated 'atr_sma_effective' column for the baseline
                atr_sma_effective_val = local_data["atr_sma_effective"].iloc[i]
                if not pd.isna(atr_sma_effective_val):
                     threshold = atr_sma_effective_val * effective_atr_multiplier
                else:
                    # Fallback if SMA is NaN (e.g., during initial window)
                    # Use current ATR * multiplier as threshold only if current ATR is not NaN
                     if not pd.isna(current_atr):
                         threshold = current_atr * effective_atr_multiplier
                     else:
                         threshold = np.nan # Keep threshold as NaN if both are NaN

            else:
                 # If no SMA period, use current ATR * multiplier if current ATR is not NaN
                 if not pd.isna(current_atr):
                     threshold = current_atr * effective_atr_multiplier
                 else:
                     threshold = np.nan # Keep threshold as NaN if current ATR is NaN

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
                # Set initial SL/TP/TSL levels based on _cleaned effective values
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
                # Set initial SL/TP/TSL levels based on _cleaned effective values
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

    # Add Rolling Sharpe Calculation
    SHARPE_ROLLING_WINDOW = 90  # Lookback window in days for rolling Sharpe
    RISK_FREE_RATE = 0.0
    ANNUALIZATION_FACTOR = 365  # Assuming daily data

    # Calculate daily returns from equity curve
    daily_returns = equity_curve.pct_change()

    # Calculate rolling mean of *excess* returns
    excess_daily_returns = daily_returns - (RISK_FREE_RATE / ANNUALIZATION_FACTOR)
    rolling_excess_mean = excess_daily_returns.rolling(window=SHARPE_ROLLING_WINDOW).mean()

    # Calculate rolling mean and std dev
    rolling_std = daily_returns.rolling(window=SHARPE_ROLLING_WINDOW).std()

    # Calculate only where std is positive to avoid division issues
    rolling_sharpe_non_annualized = pd.Series(np.nan, index=equity_curve.index)
    valid_mask = rolling_std > 0
    # Ensure alignment by using .loc with the mask for all series involved
    rolling_sharpe_non_annualized.loc[valid_mask] = rolling_excess_mean.loc[valid_mask] / rolling_std.loc[valid_mask]

    # Fill NaNs (from window or zero std) with 0 and annualize
    rolling_sharpe_annualized = (rolling_sharpe_non_annualized * np.sqrt(ANNUALIZATION_FACTOR)).fillna(0.0)

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
    equity_curve_df = pd.DataFrame({
        'equity': equity_curve,
        'rolling_sharpe': rolling_sharpe_annualized
    })

    return {
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
