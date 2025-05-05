import backtrader as bt
from .base_strategy import BaseStrategy
import backtrader.indicators as ta


class BreakoutStrategy(BaseStrategy):
    """
    Breakout Strategy using BaseStrategy features.
    Looks for price breaking out of a consolidation range, confirmed by volume and volatility.
    Uses BaseStrategy for SL/TP (if set), filters, and order management.
    """

    params = (
        # Inherits stop_loss_pct, take_profit_pct, time_window, liquidity_threshold
        ("consolidation_bars", 12),
        ("volume_spike", 3.0),  # Multiplier relative to volume MA
        ("min_volatility", 0.02),  # Min ATR as % of price
        ("volume_ma_period", 50),  # Period for baseline volume MA
        ("atr_period", 14),  # Period for ATR volatility check
        ("exit_bars", 10),  # Optional: Max bars to hold breakout before closing
    )

    def __init__(self):
        """Initializes indicators based on parameters."""
        super().__init__()

        self.highest_high = ta.Highest(self.data.high, period=self.p.consolidation_bars)
        self.lowest_low = ta.Lowest(self.data.low, period=self.p.consolidation_bars)

        self.volume_ma = ta.SMA(self.data.volume, period=self.p.volume_ma_period)
        self.atr = ta.ATR(period=self.p.atr_period)

        # self.order, self.entry_bar defined in BaseStrategy

    def next(self):
        """Implements the breakout logic with filters and order management."""
        if self.order:
            return

        # Base strategy checks
        if not self._is_time_allowed() or not self._is_liquid_enough():
            if self.position:
                self.log("Filter (Time/Liquidity) invalid, closing position.")
                self.close_position()
            return

        # --- Exit Logic --- #
        if self.position:
            # Exit after N bars (optional time-based exit)
            if (
                self.p.exit_bars is not None
                and (len(self) - self.entry_bar) >= self.p.exit_bars
            ):
                self.log(
                    f"Max holding period ({self.p.exit_bars} bars) for breakout reached. Closing position."
                )
                self.close_position()
                return
            # Exit if price falls back into range (example)
            # if self.position.size > 0 and self.data.close[0] < self.highest_high[-1]:
            #     self.log(f"Price fell back below consolidation high. Closing long.")
            #     self.close_position()
            #     return
            # elif self.position.size < 0 and self.data.close[0] > self.lowest_low[-1]:
            #     self.log(f"Price rose back above consolidation low. Closing short.")
            #     self.close_position()
            #     return

        # --- Entry Logic --- #
        if not self.position:  # Only enter if not already in a position
            # Calculate conditions for the *previous* bar (breakout occurs on current bar)
            is_volume_spike = (
                self.data.volume[0] > self.p.volume_spike * self.volume_ma[0]
            )
            current_volatility = (
                (self.atr[0] / self.data.close[0]) if self.data.close[0] != 0 else 0
            )
            is_volatile_enough = current_volatility > self.p.min_volatility

            # Long Breakout: Close breaks above the high of the consolidation period
            if self.data.close[0] > self.highest_high[-1]:
                if is_volume_spike and is_volatile_enough:
                    self.log(
                        f"LONG BREAKOUT SIGNAL: Close {self.data.close[0]:.5f} > Consol. High {self.highest_high[-1]:.5f}. Vol Spike: {is_volume_spike}, Volatility: {current_volatility*100:.2f}% > {self.p.min_volatility*100:.2f}%"
                    )
                    self.enter_long()
                else:
                    self.log(
                        f"Long breakout condition met but filters failed. Vol Spike: {is_volume_spike}, Volatility OK: {is_volatile_enough}"
                    )

            # Short Breakout: Close breaks below the low of the consolidation period
            # elif self.data.close[0] < self.lowest_low[-1]:
            #      if is_volume_spike and is_volatile_enough:
            #          self.log(f"SHORT BREAKOUT SIGNAL: Close {self.data.close[0]:.5f} < Consol. Low {self.lowest_low[-1]:.5f}. Vol Spike: {is_volume_spike}, Volatility: {current_volatility*100:.2f}% > {self.p.min_volatility*100:.2f}%")
            #          self.enter_short()
            #      else:
            #          self.log(f"Short breakout condition met but filters failed. Vol Spike: {is_volume_spike}, Volatility OK: {is_volatile_enough}")

    # notify_order is handled by BaseStrategy
