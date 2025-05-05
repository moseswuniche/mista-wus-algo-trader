import backtrader as bt
import backtrader.indicators as ta
from .base_strategy import BaseStrategy  # Import BaseStrategy


class MomentumStrategy(BaseStrategy):  # Inherit from BaseStrategy
    """
    Momentum strategy based on ATR for volatility and volume spikes.
    Uses BaseStrategy for SL/TP, filters, and order management.
    """

    params = (
        # Inherits stop_loss_pct, take_profit_pct, time_window, liquidity_threshold
        ("atr_period", 10),
        ("volume_multiplier", 2.0),  # Multiplier vs volume MA
        ("volatility_threshold", 1.5),  # Multiplier for ATR to define significant move
        ("volume_ma_period", 50),  # Added parameter
        ("exit_atr_multiplier", 0.5),  # Added parameter for exit condition
    )

    def __init__(self):
        """Initializes the strategy indicators."""
        super().__init__()  # Call base class init

        self.atr = ta.AverageTrueRange(period=self.p.atr_period)
        self.volume_ma = ta.SMA(self.data.volume, period=self.p.volume_ma_period)

        # self.order is defined in BaseStrategy

    def next(self):
        """Defines the logic for each trading bar using BaseStrategy features."""
        # Check if an order is pending
        if self.order:
            return

        # Base strategy checks
        if not self._is_time_allowed() or not self._is_liquid_enough():
            if self.position:
                self.log("Filter (Time/Liquidity) invalid, closing position.")
                self.close_position()
            return

        # Calculate conditions
        volume_spike = self.data.volume[0] > (
            self.p.volume_multiplier * self.volume_ma[0]
        )
        price_change = abs(self.data.close[0] - self.data.close[-1])
        volatility_breakout = price_change > (self.p.volatility_threshold * self.atr[0])
        exit_volatility_drop = price_change < (self.p.exit_atr_multiplier * self.atr[0])

        # --- Entry Logic --- #
        if not self.position:
            # Long entry signal: Volume spike, volatility breakout, price moving up
            if (
                volume_spike
                and volatility_breakout
                and self.data.close[0] > self.data.close[-1]
            ):
                self.log(
                    f"LONG ENTRY SIGNAL: Price Change {price_change:.4f} > Vol Thresh {self.p.volatility_threshold * self.atr[0]:.4f}, Vol Spike OK, Price Up"
                )
                self.enter_long()

            # Optional Short entry signal: Volume spike, volatility breakout, price moving down
            # elif volume_spike and volatility_breakout and self.data.close[0] < self.data.close[-1]:
            #     self.log(f"SHORT ENTRY SIGNAL: Price Change {price_change:.4f} > Vol Thresh {self.p.volatility_threshold * self.atr[0]:.4f}, Vol Spike OK, Price Down")
            #     self.enter_short()

        # --- Exit Logic --- #
        else:
            # Exit if volatility decreases significantly
            if exit_volatility_drop:
                if self.position.size > 0:
                    self.log(
                        f"LONG EXIT SIGNAL: Volatility Drop - Price Change {price_change:.4f} < Exit Thresh {self.p.exit_atr_multiplier * self.atr[0]:.4f}"
                    )
                    self.close_position()
                # elif self.position.size < 0:
                #      self.log(f"SHORT EXIT SIGNAL: Volatility Drop - Price Change {price_change:.4f} < Exit Thresh {self.p.exit_atr_multiplier * self.atr[0]:.4f}")
                #      self.close_position()

    # notify_order is handled by BaseStrategy
