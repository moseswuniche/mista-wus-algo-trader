import backtrader as bt
from .base_strategy import BaseStrategy
import backtrader.indicators as ta


class HybridStrategy(BaseStrategy):
    """
    Hybrid Strategy using BaseStrategy features.
    Combines EMA crossover and MACD signals for entries and exits.
    Uses BaseStrategy for SL/TP (if set), filters, and order management.
    """

    params = (
        # Inherits stop_loss_pct, take_profit_pct, time_window, liquidity_threshold
        ("ema_fast", 7),
        ("ema_slow", 21),
        ("macd_fast", 5),
        ("macd_slow", 13),
        ("macd_signal", 4),
    )

    def __init__(self):
        """Initializes indicators based on parameters."""
        super().__init__()

        self.ema_fast = ta.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = ta.EMA(self.data.close, period=self.p.ema_slow)
        self.macd = ta.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal,
        )

        # self.order defined in BaseStrategy

    def next(self):
        """Implements the hybrid EMA/MACD logic with filters and order management."""
        if self.order:
            return

        # Base strategy checks
        if not self._is_time_allowed() or not self._is_liquid_enough():
            if self.position:
                self.log("Filter (Time/Liquidity) invalid, closing position.")
                self.close_position()
            return

        # --- Signal Calculation --- #
        ema_long_signal = self.ema_fast[0] > self.ema_slow[0]
        macd_long_signal = self.macd.lines.macd[0] > self.macd.lines.signal[0]
        long_entry_condition = ema_long_signal and macd_long_signal

        ema_short_signal = self.ema_fast[0] < self.ema_slow[0]
        macd_short_signal = self.macd.lines.macd[0] < self.macd.lines.signal[0]
        short_entry_condition = ema_short_signal and macd_short_signal

        # --- Entry Logic --- #
        if not self.position:
            if long_entry_condition:
                self.log(
                    f"LONG ENTRY SIGNAL: EMA ({self.ema_fast[0]:.5f} > {self.ema_slow[0]:.5f}) AND MACD ({self.macd.lines.macd[0]:.5f} > {self.macd.lines.signal[0]:.5f})"
                )
                self.enter_long()  # Base handles SL/TP if params are set

            # Optional Short Entry
            # elif short_entry_condition:
            #     self.log(f"SHORT ENTRY SIGNAL: EMA ({self.ema_fast[0]:.5f} < {self.ema_slow[0]:.5f}) AND MACD ({self.macd.lines.macd[0]:.5f} < {self.macd.lines.signal[0]:.5f})")
            #     self.enter_short()

        # --- Exit Logic --- #
        else:
            # Exit long if either EMA or MACD signal reverses
            if self.position.size > 0 and (not ema_long_signal or not macd_long_signal):
                self.log(
                    f"LONG EXIT SIGNAL: EMA Long={ema_long_signal}, MACD Long={macd_long_signal}"
                )
                self.close_position()

            # Optional Short Exit
            # elif self.position.size < 0 and (not ema_short_signal or not macd_short_signal):
            #      self.log(f"SHORT EXIT SIGNAL: EMA Short={ema_short_signal}, MACD Short={macd_short_signal}")
            #      self.close_position()

    # notify_order is handled by BaseStrategy
