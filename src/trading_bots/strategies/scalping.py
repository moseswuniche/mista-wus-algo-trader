import backtrader as bt
import backtrader.indicators as ta
from .base_strategy import BaseStrategy  # Import BaseStrategy


class ScalpingStrategy(BaseStrategy):  # Inherit from BaseStrategy
    """
    Scalping strategy based on EMA crossover, RSI, and volume spikes.
    Uses BaseStrategy for SL/TP, filters, and order management.
    """

    params = (
        # Inherits stop_loss_pct, take_profit_pct, time_window, liquidity_threshold
        ("ema_fast", 5),
        ("ema_slow", 21),
        ("rsi_period", 7),
        ("rsi_overbought", 70),  # Added threshold parameter
        ("volume_spike_multiplier", 1.5),
        ("volume_ma_period", 50),  # Added parameter
        # min_liquidity is handled by BaseStrategy liquidity_threshold
    )

    def __init__(self):
        """Initializes the strategy indicators."""
        super().__init__()  # Call base class init

        self.ema_fast = ta.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = ta.EMA(self.data.close, period=self.p.ema_slow)
        self.rsi = ta.RSI(self.data.close, period=self.p.rsi_period)
        self.volume_ma = ta.SMA(self.data.volume, period=self.p.volume_ma_period)

        # self.order is defined in BaseStrategy

    def next(self):
        """Defines the logic for each trading bar using BaseStrategy features."""
        # Check if an order is pending
        if self.order:
            return

        # Base strategy checks for time window and liquidity
        # Note: liquidity_threshold param from BaseStrategy is used by _is_liquid_enough()
        if not self._is_time_allowed() or not self._is_liquid_enough():
            if self.position:
                self.log("Filter (Time/Liquidity) invalid, closing position.")
                self.close_position()
            return

        # Calculate volume spike condition
        volume_spike_condition = self.data.volume[0] > (
            self.p.volume_spike_multiplier * self.volume_ma[0]
        )

        # Define long entry signal
        long_signal = (
            self.ema_fast[0] > self.ema_slow[0]
            and self.rsi[0] < self.p.rsi_overbought
            and volume_spike_condition
        )

        # Define short entry signal (Example - mirror logic)
        # rsi_oversold = 100 - self.p.rsi_overbought # Example symmetric threshold
        # short_signal = (
        #     self.ema_fast[0] < self.ema_slow[0] and
        #     self.rsi[0] > rsi_oversold and
        #     volume_spike_condition
        # )

        # --- Entry and Exit Logic --- #
        if not self.position:
            if long_signal:
                self.log(
                    f"LONG ENTRY SIGNAL: EMA Fast {self.ema_fast[0]:.5f} > Slow {self.ema_slow[0]:.5f}, RSI {self.rsi[0]:.2f} < {self.p.rsi_overbought}, Vol Spike OK"
                )
                self.enter_long()
            # elif short_signal: # Enable if shorting is desired
            #     self.log(f"SHORT ENTRY SIGNAL: EMA Fast {self.ema_fast[0]:.5f} < Slow {self.ema_slow[0]:.5f}, RSI {self.rsi[0]:.2f} > {rsi_oversold}, Vol Spike OK")
            #     self.enter_short()
        else:
            # Simple exit logic: Exit if EMA crossover reverses
            if self.position.size > 0 and self.ema_fast[0] < self.ema_slow[0]:
                self.log(
                    f"LONG EXIT SIGNAL: EMA Fast {self.ema_fast[0]:.5f} < Slow {self.ema_slow[0]:.5f}"
                )
                self.close_position()
            # elif self.position.size < 0 and self.ema_fast[0] > self.ema_slow[0]: # For short positions
            #      self.log(f"SHORT EXIT SIGNAL: EMA Fast {self.ema_fast[0]:.5f} > Slow {self.ema_slow[0]:.5f}")
            #      self.close_position()

    # notify_order is handled by BaseStrategy
