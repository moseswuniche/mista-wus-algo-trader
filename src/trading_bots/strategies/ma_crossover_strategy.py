import backtrader as bt
from .base_strategy import BaseStrategy
import backtrader.indicators as ta


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy using BaseStrategy features.
    Enters long when fast MA crosses above slow MA, exits when it crosses below.
    Applies trend filter, time window, and liquidity checks from BaseStrategy.
    Uses SL/TP defined in parameters via BaseStrategy bracket orders.
    """

    params = (
        # Inherits stop_loss_pct, take_profit_pct, time_window, liquidity_threshold
        ("fast_period", 5),
        ("slow_period", 21),
        ("ma_type", "EMA"),
        ("trend_filter_period", 75),
    )

    def __init__(self):
        """Initializes indicators based on parameters."""
        # Call base class __init__ first
        super().__init__()

        ma_indicator = ta.EMA if self.p.ma_type == "EMA" else ta.SMA

        self.ma_fast = ma_indicator(self.data.close, period=self.p.fast_period)
        self.ma_slow = ma_indicator(self.data.close, period=self.p.slow_period)
        self.crossover = ta.CrossOver(self.ma_fast, self.ma_slow)

        if self.p.trend_filter_period is not None and self.p.trend_filter_period > 0:
            self.trend_filter = ma_indicator(
                self.data.close, period=self.p.trend_filter_period
            )
        else:
            self.trend_filter = None

        # self.order is already defined in BaseStrategy

    def next(self):
        """Implements the MA crossover logic with filters and order management."""
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

        # Trend filter check
        if self.trend_filter is not None:
            if (
                self.data.close[0] < self.trend_filter[0]
            ):  # Only allow longs if above trend MA
                # If in a long position initiated when trend was valid, close it now
                if self.position.size > 0:
                    self.log(
                        f"Trend filter ({self.p.trend_filter_period} MA) turned negative, closing long position."
                    )
                    self.close_position()
                return  # Skip long entry signals
            # Add short logic if needed: block shorts if price > trend_filter?

        # --- Entry and Exit Logic ---
        if not self.position:
            if self.crossover[0] > 0:  # Fast crosses above slow
                self.log(
                    f"LONG ENTRY SIGNAL: Fast MA {self.ma_fast[0]:.5f} > Slow MA {self.ma_slow[0]:.5f}"
                )
                self.enter_long()  # Use base class method for entry with SL/TP

        # If in a long position
        elif self.position.size > 0:
            if self.crossover[0] < 0:  # Fast crosses below slow
                self.log(
                    f"LONG EXIT SIGNAL: Fast MA {self.ma_fast[0]:.5f} < Slow MA {self.ma_slow[0]:.5f}"
                )
                self.close_position()  # Use base class method for closing

    # notify_order is handled by BaseStrategy
