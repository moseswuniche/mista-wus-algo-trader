import backtrader as bt
from .base_strategy import BaseStrategy
import backtrader.indicators as ta


class BollingerBandReversionStrategy(BaseStrategy):
    """
    Bollinger Band Mean Reversion Strategy using BaseStrategy features.
    Enters long when price crosses below lower band, exits near middle band.
    Applies time window and liquidity checks from BaseStrategy.
    Uses SL/TP defined in parameters via BaseStrategy bracket orders.
    """

    params = (
        # Inherits stop_loss_pct, take_profit_pct, time_window, liquidity_threshold
        ("bb_period", 15),
        ("bb_std_dev", 1.8),
        # stop_loss_pct, take_profit_pct, liquidity_threshold are used by BaseStrategy
    )

    def __init__(self):
        """Initializes indicators based on parameters."""
        # Call base class __init__ first
        super().__init__()

        self.bbands = ta.BollingerBands(
            period=self.p.bb_period, devfactor=self.p.bb_std_dev
        )

        # self.order is defined in BaseStrategy

    def next(self):
        """Implements the BB reversion logic with filters and order management."""
        # Check if an order is pending from the base class
        if self.order:
            return

        # Base strategy checks for time window and liquidity
        if not self._is_time_allowed() or not self._is_liquid_enough():
            # Close position if filters become invalid while holding
            if self.position:
                self.log("Filter (Time/Liquidity) invalid, closing position.")
                self.close_position()
            return  # Skip signal generation

        # --- Entry and Exit Logic ---
        if not self.position:
            if (
                self.data.close[0] < self.bbands.lines.bot[0]
            ):  # Price closes below lower band
                self.log(
                    f"LONG ENTRY SIGNAL: Close {self.data.close[0]:.5f} < Lower BB {self.bbands.lines.bot[0]:.5f}"
                )
                self.enter_long()  # Use base class method for entry with SL/TP

            # Optional: Add short entry if price closes above upper band
            # elif self.data.close[0] > self.bbands.lines.top[0]:
            #     self.log(f'SHORT ENTRY SIGNAL: Close {self.data.close[0]:.5f} > Upper BB {self.bbands.lines.top[0]:.5f}')
            #     self.enter_short()

        else:  # In a position, check for exit
            # Exit when price reverts towards the middle band (example)
            if self.position.size > 0 and self.data.close[0] > self.bbands.lines.mid[0]:
                self.log(
                    f"LONG EXIT SIGNAL: Close {self.data.close[0]:.5f} > Mid BB {self.bbands.lines.mid[0]:.5f}"
                )
                self.close_position()  # Use base class method for closing

            # Optional: Add short exit logic
            # elif self.position.size < 0 and self.data.close[0] < self.bbands.lines.mid[0]:
            #      self.log(f'SHORT EXIT SIGNAL: Close {self.data.close[0]:.5f} < Mid BB {self.bbands.lines.mid[0]:.5f}')
            #      self.close_position()

    # notify_order is handled by BaseStrategy
