import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

from .strategies.base_strategy import Strategy

# Get logger instance
logger = logging.getLogger(__name__)

def run_backtest(data: pd.DataFrame, strategy: Strategy, units: float, initial_balance: float = 10000.0, commission_bps: float = 0.0) -> Dict[str, Any]:
    """
    Runs a simple vectorised backtest for a given strategy.

    Args:
        data: DataFrame with historical OHLCV data, indexed by Date.
        strategy: An instantiated strategy object.
        units: The fixed amount of the asset to trade per signal.
        initial_balance: Starting balance for calculating returns.
        commission_bps: Commission fee in basis points (e.g., 10 for 0.1%).

    Returns:
        A dictionary containing backtest results:
        - 'cumulative_profit': Total profit/loss in quote currency.
        - 'final_balance': The ending balance.
        - 'total_trades': Number of closing trades executed.
        - 'win_rate': Percentage of winning trades (if any trades occurred).
        - 'performance_summary': DataFrame of trades.
    """
    if 'Close' not in data.columns:
        logger.error("Data must contain 'Close' column for backtest.")
        raise ValueError("Data must contain 'Close' column.")

    logger.info(f"Running backtest for strategy: {strategy.__class__.__name__} with params: {strategy.params}, Commission: {commission_bps} bps")

    # 1. Generate Signals
    signals = strategy.generate_signals(data)

    # Ensure signals align with data index
    signals = signals.reindex(data.index).fillna(0).astype(int)

    # 2. Simulate Positions and Trades
    # Shift signals by 1 to avoid lookahead bias (trade on the next bar's info)
    # We decide position based on the signal from the *previous* bar close.
    positions = signals.shift(1).fillna(0)

    # Calculate changes in position to identify trades
    trades = positions.diff().fillna(0)

    # 3. Calculate P&L (Modified for detailed logging and commission)
    trade_log = []
    current_pos = 0
    entry_price = 0.0
    entry_time = None
    trade_count = 0
    winning_trades = 0
    balance = initial_balance
    commission_rate = commission_bps / 10000.0 # Convert basis points to rate

    # Iterate through bars where a position *could* change
    for i, trade_signal in enumerate(trades):
        timestamp = data.index[i]
        price = data['Close'].iloc[i]

        if pd.isna(price):
             logger.warning(f"NaN price found at index {i}, timestamp {timestamp}. Skipping bar.")
             continue
             
        # Check if we need to exit the current position
        exit_trade = False
        if current_pos != 0: # If we are currently in a position
            # Exit if signal flips or goes neutral
            if positions.iloc[i] != current_pos: 
                 exit_trade = True
        
        if exit_trade:
            exit_price = price
            exit_time = timestamp
            gross_profit = (exit_price - entry_price) * units * current_pos
            commission_exit = abs(exit_price * units * commission_rate)
            net_profit = gross_profit - commission_exit
            
            balance += net_profit # Update balance immediately on exit
            trade_count += 1
            if net_profit > 0:
                winning_trades += 1
            
            if trade_log: # Update the last opened trade log entry
                trade_log[-1].update({
                    'Exit Time': exit_time,
                    'Exit Price': exit_price,
                    'Commission': commission_exit, # Add commission
                    'Profit/Loss': net_profit # Log net profit
                })
            logger.debug(f"EXIT Trade: {timestamp} - Pos: {current_pos} -> 0 | Entry: {entry_price:.2f} | Exit: {exit_price:.2f} | Comm: {commission_exit:.4f} | PnL: {net_profit:.2f} | Balance: {balance:.2f}")
            current_pos = 0 # We are now flat
            entry_price = 0.0
            entry_time = None

        # Check if we need to enter a new position
        if positions.iloc[i] != 0 and current_pos == 0: # If signal is non-neutral and we are flat
            entry_price = price
            entry_time = timestamp
            current_pos = positions.iloc[i]
            commission_entry = abs(entry_price * units * commission_rate)
            balance -= commission_entry # Deduct entry commission from balance
            
            trade_log.append({
                'Entry Time': entry_time,
                'Entry Price': entry_price,
                'Position': 'Long' if current_pos == 1 else 'Short',
                'Commission': commission_entry, # Add commission
                'Exit Time': None,
                'Exit Price': None,
                'Profit/Loss': None
            })
            logger.debug(f"ENTRY Trade: {timestamp} - Pos: 0 -> {current_pos} | Entry: {entry_price:.2f} | Comm: {commission_entry:.4f} | Balance: {balance:.2f}")

    # 4. Final Performance Metrics
    performance_summary = pd.DataFrame(trade_log)
    if not performance_summary.empty:
        if 'Entry Time' in performance_summary.columns:
            performance_summary.set_index('Entry Time', inplace=True)
        if 'Profit/Loss' not in performance_summary.columns:
            performance_summary['Profit/Loss'] = np.nan
            
    total_closed_trades = performance_summary['Profit/Loss'].count() if 'Profit/Loss' in performance_summary.columns else 0
    win_rate = (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0

    # Use the sum of logged profits for final calculation
    logged_total_profit = performance_summary['Profit/Loss'].sum() if total_closed_trades > 0 else 0
    final_balance = initial_balance + logged_total_profit - performance_summary['Commission'].sum() # Ensure all commission is accounted for

    logger.info(f"Backtest complete. Final Balance: {final_balance:.2f}, Total Profit: {logged_total_profit:.2f}, Trades: {total_closed_trades}, Win Rate: {win_rate:.2f}%")

    return {
        "cumulative_profit": logged_total_profit,
        "final_balance": final_balance,
        "total_trades": total_closed_trades,
        "win_rate": win_rate,
        "performance_summary": performance_summary
    } 