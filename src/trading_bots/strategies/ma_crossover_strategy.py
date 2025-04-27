import pandas as pd
from typing import Tuple

from .base_strategy import Strategy

class MovingAverageCrossoverStrategy(Strategy):
    """
    A trend-following strategy based on the crossover of two Exponential Moving Averages (EMAs).
    Goes long when the fast EMA crosses above the slow EMA.
    Goes short when the fast EMA crosses below the slow EMA.
    """
    def __init__(self, fast_period: int = 9, slow_period: int = 21) -> None:
        """
        Initializes the MovingAverageCrossoverStrategy.

        Args:
            fast_period: The lookback period for the fast EMA.
            slow_period: The lookback period for the slow EMA.
        """
        if not isinstance(fast_period, int) or not isinstance(slow_period, int) or fast_period <= 0 or slow_period <= 0:
            raise ValueError("EMA periods must be positive integers.")
        if fast_period >= slow_period:
            raise ValueError("Fast EMA period must be less than Slow EMA period.")

        super().__init__(fast_period=fast_period, slow_period=slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculates Exponential Moving Average."""
        # Using adjust=False aligns better with many trading platforms' EMA calculations
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on EMA crossovers.

        Args:
            data: DataFrame containing at least the 'Close' price column.

        Returns:
            A pandas Series containing the position signal (1 for long, -1 for short, 0 for neutral).
        """
        if 'Close' not in data.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column.")

        close_prices = data['Close']

        # Calculate EMAs
        ema_fast = self._calculate_ema(close_prices, self.fast_period)
        ema_slow = self._calculate_ema(close_prices, self.slow_period)

        # Determine signal based on crossover
        # Initialize positions with 0 (neutral)
        position = pd.Series(index=data.index, data=0, dtype=int)

        # Go long if fast EMA is above slow EMA
        position[ema_fast > ema_slow] = 1

        # Go short if fast EMA is below slow EMA
        position[ema_fast < ema_slow] = -1

        # Signals are only valid after both EMAs have enough data
        first_valid_index = max(self.fast_period, self.slow_period)
        position[:first_valid_index] = 0 # Set initial positions before enough data to 0

        # Fill any remaining NaNs (shouldn't be any with min_periods, but just in case)
        position.fillna(0, inplace=True)

        return position.astype(int) 