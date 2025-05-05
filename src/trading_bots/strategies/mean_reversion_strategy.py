import backtrader as bt
from .base_strategy import BaseStrategy
import backtrader.indicators as ta


# Custom ZScore Indicator
class ZScore(bt.Indicator):
    lines = ("zscore",)
    params = (("period", 20),)

    def __init__(self):
        self.mean = ta.SMA(self.data, period=self.p.period)
        self.std = ta.StdDev(self.data, period=self.p.period)

    def next(self):
        if self.std[0] == 0:  # Avoid division by zero
            self.lines.zscore[0] = 0.0
        else:
            self.lines.zscore[0] = (self.data[0] - self.mean[0]) / self.std[0]


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Z-Score and RSI.
    Enters long on low Z-Score/RSI, short on high Z-Score/RSI.
    Exits on profit target, max holding period, or reversion signal.
    Uses BaseStrategy for SL/TP (if set), filters, and order management.
    """

    params = (
        # Inherits stop_loss_pct, take_profit_pct, time_window, liquidity_threshold
        ("z_score_period", 20),
        ("rsi_period", 5),
        ("profit_target", 0.015),
        ("max_holding_period", 45),
        ("rsi_oversold", 30),  # Added thresholds
        ("rsi_overbought", 70),
        ("zscore_low_threshold", -2.0),  # Added thresholds
        ("zscore_high_threshold", 2.0),
    )

    def __init__(self):
        """Initializes indicators based on parameters."""
        super().__init__()

        self.z_score = ZScore(self.data.close, period=self.p.z_score_period)
        self.rsi = ta.RSI(self.data.close, period=self.p.rsi_period)

        # self.order, self.entry_bar defined in BaseStrategy

    def next(self):
        """Implements the mean reversion logic with filters and order management."""
        if self.order:
            return

        # Base strategy checks
        if not self._is_time_allowed() or not self._is_liquid_enough():
            if self.position:
                self.log("Filter (Time/Liquidity) invalid, closing position.")
                self.close_position()
            return

        # --- Exit Logic ---
        if self.position:
            current_profit = 0
            if self.position.size > 0:  # Long position exit checks
                current_profit = (
                    self.data.close[0] - self.position.price
                ) / self.position.price
                # Profit Target Exit
                if (
                    self.p.profit_target is not None
                    and current_profit >= self.p.profit_target
                ):
                    self.log(
                        f"Profit target reached ({current_profit*100:.2f}%). Closing long."
                    )
                    self.close_position()
                    return
                # Max Holding Period Exit
                elif (
                    self.p.max_holding_period is not None
                    and (len(self) - self.entry_bar) >= self.p.max_holding_period
                ):
                    self.log(
                        f"Max holding period ({self.p.max_holding_period} bars) reached. Closing long."
                    )
                    self.close_position()
                    return
                # Reversion Exit (e.g., Z-Score goes back above 0)
                # elif self.z_score[0] > 0:
                #      self.log(f"Z-Score reverted ({self.z_score[0]:.2f}). Closing long.")
                #      self.close_position()
                #      return

            elif self.position.size < 0:  # Short position exit checks
                current_profit = (
                    self.position.price - self.data.close[0]
                ) / self.position.price  # Note reversal for short
                # Profit Target Exit
                if (
                    self.p.profit_target is not None
                    and current_profit >= self.p.profit_target
                ):
                    self.log(
                        f"Profit target reached ({current_profit*100:.2f}%). Closing short."
                    )
                    self.close_position()
                    return
                # Max Holding Period Exit
                elif (
                    self.p.max_holding_period is not None
                    and (len(self) - self.entry_bar) >= self.p.max_holding_period
                ):
                    self.log(
                        f"Max holding period ({self.p.max_holding_period} bars) reached. Closing short."
                    )
                    self.close_position()
                    return
                # Reversion Exit (e.g., Z-Score goes back below 0)
                # elif self.z_score[0] < 0:
                #      self.log(f"Z-Score reverted ({self.z_score[0]:.2f}). Closing short.")
                #      self.close_position()
                #      return

        # --- Entry Logic ---
        if not self.position:  # Only enter if not already in a position
            # Long entry condition: RSI oversold AND Z-score low
            if (
                self.rsi[0] < self.p.rsi_oversold
                and self.z_score[0] < self.p.zscore_low_threshold
            ):
                self.log(
                    f"LONG ENTRY SIGNAL: RSI {self.rsi[0]:.2f} < {self.p.rsi_oversold}, Z-Score {self.z_score[0]:.2f} < {self.p.zscore_low_threshold}"
                )
                self.enter_long()

            # Short entry condition: RSI overbought AND Z-score high
            # elif self.rsi[0] > self.p.rsi_overbought and self.z_score[0] > self.p.zscore_high_threshold:
            #      self.log(f"SHORT ENTRY SIGNAL: RSI {self.rsi[0]:.2f} > {self.p.rsi_overbought}, Z-Score {self.z_score[0]:.2f} > {self.p.zscore_high_threshold}")
            #      self.enter_short()

    # notify_order is handled by BaseStrategy
