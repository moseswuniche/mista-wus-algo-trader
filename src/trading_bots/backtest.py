import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from .strategies.base_strategy import Strategy

# Get logger instance
logger = logging.getLogger(__name__)

# --- Performance Metrics Calculations ---
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 365) -> float:
    """Calculates the Sharpe ratio from a pandas Series of returns."""
    if returns.empty or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / periods_per_year
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculates the maximum drawdown from an equity curve."""
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()
    return abs(max_drawdown) # Return as a positive percentage

def calculate_profit_factor(performance_summary: pd.DataFrame) -> float:
     """Calculates the profit factor from the performance summary DataFrame."""
     if performance_summary.empty or 'Profit/Loss' not in performance_summary.columns:
         return 0.0
     gross_profits = performance_summary[performance_summary['Profit/Loss'] > 0]['Profit/Loss'].sum()
     gross_losses = abs(performance_summary[performance_summary['Profit/Loss'] < 0]['Profit/Loss'].sum())
     if gross_losses == 0:
         return float('inf') if gross_profits > 0 else 0.0 # Avoid division by zero
     return gross_profits / gross_losses
     
# --- Main Backtest Function ---
def run_backtest(
    data: pd.DataFrame, 
    strategy: Strategy, 
    units: float, 
    initial_balance: float = 10000.0, 
    commission_bps: float = 0.0,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    trailing_stop_loss_pct: Optional[float] = None
) -> Dict[str, Any]:
    """
    Runs a bar-by-bar backtest simulating intra-bar SL/TP/TSL checks.

    Args:
        data: DataFrame with historical OHLCV data, indexed by Date.
              Must include 'Open', 'High', 'Low', 'Close' columns.
        strategy: An instantiated strategy object.
        units: The fixed amount of the asset to trade per signal.
        initial_balance: Starting balance for calculating returns.
        commission_bps: Commission fee in basis points (e.g., 10 for 0.1%).
        stop_loss_pct: Optional fixed stop loss percentage (e.g., 0.02 for 2%).
        take_profit_pct: Optional fixed take profit percentage (e.g., 0.04 for 4%).
        trailing_stop_loss_pct: Optional trailing stop loss percentage (e.g., 0.01 for 1%).

    Returns:
        A dictionary containing backtest results including:
        - 'cumulative_profit': Total profit/loss in quote currency.
        - 'final_balance': The ending balance.
        - 'total_trades': Number of closing trades executed.
        - 'win_rate': Percentage of winning trades (if any trades occurred).
        - 'sharpe_ratio': Annualized Sharpe Ratio.
        - 'max_drawdown': Maximum Drawdown percentage.
        - 'profit_factor': Profit Factor.
        - 'performance_summary': DataFrame of trades.
        - 'equity_curve': Pandas Series representing portfolio value over time.
    """
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        logger.error(f"Data must contain columns: {required_cols} for backtest with SL/TP.")
        raise ValueError(f"Data must contain columns: {required_cols}.")

    logger.info(f"Running backtest for: {strategy.__class__.__name__}")
    logger.info(f"Params: {strategy.params}, SL: {stop_loss_pct}, TP: {take_profit_pct}, TSL: {trailing_stop_loss_pct}, Comm: {commission_bps} bps")

    # 1. Generate Signals (still needed for entry/exit triggers)
    signals = strategy.generate_signals(data)
    signals = signals.reindex(data.index).fillna(0).astype(int)

    # 2. Initialize Backtest State
    trade_log: List[Dict[str, Any]] = []
    current_pos = 0 # Position: 1=Long, -1=Short, 0=Flat
    entry_price = 0.0
    entry_time = None
    trade_count = 0 # Number of closed trades
    winning_trades = 0
    balance = initial_balance
    commission_rate = commission_bps / 10000.0 
    equity_curve = pd.Series(index=data.index, dtype=float)
    equity_curve.iloc[0] = initial_balance
    
    stop_loss_level: Optional[float] = None
    take_profit_level: Optional[float] = None
    tsl_peak_price: Optional[float] = None
    # last_trade_log_entry: Optional[Dict[str, Any]] = None # Not strictly needed with append/update approach

    # Iterate through bars - Start from 1 to allow looking at previous close
    for i in range(1, len(data)):
        # --- Get current and previous bar data --- 
        timestamp = data.index[i]
        prev_timestamp = data.index[i-1]
        
        open_price = data['Open'].iloc[i]
        high_price = data['High'].iloc[i]
        low_price = data['Low'].iloc[i]
        close_price = data['Close'].iloc[i]
        prev_close_price = data['Close'].iloc[i-1]

        # Forward fill equity from previous day before calculating changes
        if pd.isna(equity_curve.iloc[i]):
            equity_curve.iloc[i] = equity_curve.iloc[i-1]
            
        # Handle potential NaN prices in data
        if pd.isna(close_price) or pd.isna(prev_close_price) or pd.isna(high_price) or pd.isna(low_price):
             logger.warning(f"NaN price detected at {timestamp}. Skipping bar, carrying forward equity.")
             continue

        # --- Calculate Unrealized PnL and Update Equity Curve --- 
        unrealized_pnl_change = 0
        if current_pos != 0:
             unrealized_pnl_change = (close_price - prev_close_price) * units * current_pos
        equity_curve.iloc[i] += unrealized_pnl_change
        
        # --- Process Active Position (Check Exits) --- 
        exit_triggered_this_bar = False
        if current_pos != 0:
            exit_price = 0.0
            exit_reason = ""
            commission_exit = 0.0
            net_profit = 0.0
            
            # (A) Check Take Profit Hit (using High/Low)
            if take_profit_level is not None:
                if current_pos == 1 and high_price >= take_profit_level:
                    exit_price = take_profit_level # Assume execution at TP level
                    exit_reason = "Take Profit"
                    exit_triggered_this_bar = True
                elif current_pos == -1 and low_price <= take_profit_level:
                    exit_price = take_profit_level
                    exit_reason = "Take Profit"
                    exit_triggered_this_bar = True

            # (B) Check Stop Loss Hit (using High/Low, after TSL update)
            if not exit_triggered_this_bar and stop_loss_level is not None: 
                # Update TSL first
                if trailing_stop_loss_pct is not None and tsl_peak_price is not None:
                    initial_sl = stop_loss_level
                    potential_tsl = None
                    if current_pos == 1:
                        tsl_peak_price = max(tsl_peak_price, high_price)
                        potential_tsl = tsl_peak_price * (1 - trailing_stop_loss_pct)
                        stop_loss_level = max(initial_sl, potential_tsl) if potential_tsl > entry_price else initial_sl
                    elif current_pos == -1:
                        tsl_peak_price = min(tsl_peak_price, low_price)
                        potential_tsl = tsl_peak_price * (1 + trailing_stop_loss_pct)
                        stop_loss_level = min(initial_sl, potential_tsl) if potential_tsl < entry_price else initial_sl
                    
                    if stop_loss_level != initial_sl:
                         logger.debug(f"TSL Update at {timestamp}: Peak: {tsl_peak_price:.5f} -> New SL: {stop_loss_level:.5f}")
                
                # Now check SL hit
                if current_pos == 1 and low_price <= stop_loss_level:
                    exit_price = stop_loss_level # Assume execution at SL level
                    exit_reason = "Stop Loss / TSL"
                    exit_triggered_this_bar = True
                elif current_pos == -1 and high_price >= stop_loss_level:
                    exit_price = stop_loss_level
                    exit_reason = "Stop Loss / TSL"
                    exit_triggered_this_bar = True
                    
            # (C) Check Strategy Exit Signal (using Close)
            if not exit_triggered_this_bar:
                # Exit if signal flips or goes neutral for the *current* bar
                if signals.iloc[i] != current_pos:
                    exit_price = close_price # Exit at close based on signal
                    exit_reason = "Strategy Signal"
                    exit_triggered_this_bar = True
            
            # --- Process Exit --- 
            if exit_triggered_this_bar:
                gross_profit = (exit_price - entry_price) * units * current_pos
                commission_exit = abs(exit_price * units * commission_rate)
                net_profit = gross_profit - commission_exit
                
                balance += net_profit # Update cash balance
                equity_curve.iloc[i] -= commission_exit # Reflect commission in equity
                trade_count += 1
                if net_profit > 0:
                    winning_trades += 1
                
                # Find the corresponding entry log and update it
                # This assumes trades don't overlap in the log list approach
                found_entry = False
                for log in reversed(trade_log):
                     if log.get('Position') == ('Long' if current_pos == 1 else 'Short') and log.get('Exit Time') is None:
                         log.update({
                             'Exit Time': timestamp,
                             'Exit Price': exit_price,
                             'Commission Exit': commission_exit,
                             'Profit/Loss': net_profit,
                             'Exit Reason': exit_reason
                         })
                         found_entry = True
                         break
                if not found_entry:
                     logger.error(f"Could not find matching entry log to update for exit at {timestamp}")

                logger.debug(f"EXIT Trade ({exit_reason}): {timestamp} - Pos: {current_pos} -> 0 | Entry: {entry_price:.2f} | Exit: {exit_price:.2f} | Comm: {commission_exit:.4f} | PnL: {net_profit:.2f} | Balance: {balance:.2f} | Equity: {equity_curve.iloc[i]:.2f}")
                
                # Reset state after exit
                current_pos = 0 
                entry_price = 0.0
                entry_time = None
                stop_loss_level = None
                take_profit_level = None
                tsl_peak_price = None
        
        # --- Process Entry Signal (only if flat and no exit occurred this bar) --- 
        if not exit_triggered_this_bar and current_pos == 0: 
            signal_this_bar = signals.iloc[i]
            if signal_this_bar != 0: # If signal is non-neutral and we are flat
                entry_price = close_price # Enter at close of the signal bar
                entry_time = timestamp
                current_pos = signal_this_bar # 1 for Long, -1 for Short
                commission_entry = abs(entry_price * units * commission_rate)
                
                balance -= commission_entry # Deduct entry commission from cash balance
                equity_curve.iloc[i] -= commission_entry # Reflect commission in equity
                
                # Calculate initial SL/TP
                if stop_loss_pct:
                    stop_loss_level = entry_price * (1 - stop_loss_pct * current_pos)
                    logger.debug(f"Initial SL set: {stop_loss_level:.5f}")
                else: stop_loss_level = None
                
                if take_profit_pct:
                    take_profit_level = entry_price * (1 + take_profit_pct * current_pos)
                    logger.debug(f"Initial TP set: {take_profit_level:.5f}")
                else: take_profit_level = None
                    
                # Initialize TSL peak
                tsl_peak_price = entry_price if trailing_stop_loss_pct else None
                if tsl_peak_price: logger.debug(f"Initial TSL Peak set: {tsl_peak_price:.5f}")

                # Add new entry to log
                trade_log.append({
                    'Entry Time': entry_time,
                    'Entry Price': entry_price,
                    'Position': 'Long' if current_pos == 1 else 'Short',
                    'Commission Entry': commission_entry,
                    'Exit Time': None,
                    'Exit Price': None,
                    'Commission Exit': None,
                    'Profit/Loss': None,
                    'Exit Reason': None
                })
                logger.debug(f"ENTRY Trade: {timestamp} - Pos: 0 -> {current_pos} | Entry: {entry_price:.2f} | Comm: {commission_entry:.4f} | Balance: {balance:.2f} | Equity: {equity_curve.iloc[i]:.2f}")

    # --- End of Loop --- 

    # Final Performance Metrics (recalculate based on updated log)
    performance_summary = pd.DataFrame(trade_log)
    if not performance_summary.empty:
        performance_summary.set_index('Entry Time', inplace=True, drop=False) # Keep Entry Time as col too
        total_closed_trades = performance_summary['Profit/Loss'].count()
        winning_trades = performance_summary[performance_summary['Profit/Loss'] > 0].shape[0]
        logged_total_profit = performance_summary['Profit/Loss'].sum()
        total_commission = performance_summary['Commission Entry'].sum() + performance_summary['Commission Exit'].fillna(0).sum()
    else:
        total_closed_trades = 0
        winning_trades = 0
        logged_total_profit = 0.0
        total_commission = 0.0
        
    win_rate = (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
    final_balance = initial_balance + logged_total_profit # Logged profit is already net of commission

    # Calculate additional metrics
    daily_returns = equity_curve.pct_change().dropna()
    sharpe_ratio = calculate_sharpe_ratio(daily_returns) # Assumes daily data for periods_per_year
    # Adjust periods_per_year if using non-daily data
    max_drawdown = calculate_max_drawdown(equity_curve)
    profit_factor = calculate_profit_factor(performance_summary)

    logger.info(f"Backtest complete. Final Balance: {final_balance:.2f}, Total Net Profit: {logged_total_profit:.2f}, Total Commission: {total_commission:.2f}, Trades: {total_closed_trades}, Win Rate: {win_rate:.2f}%")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown*100:.2f}%")
    logger.info(f"Profit Factor: {profit_factor:.2f}")

    # Fill any remaining NaNs in equity curve
    equity_curve = equity_curve.ffill().fillna(initial_balance)

    return {
        "cumulative_profit": logged_total_profit,
        "final_balance": final_balance,
        "total_trades": total_closed_trades,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "performance_summary": performance_summary,
        "equity_curve": equity_curve
    } 