import pandas as pd
from typing import Tuple

from .base_strategy import Strategy

class RsiMeanReversionStrategy(Strategy):
    """
    A mean-reversion strategy based on the Relative Strength Index (RSI).
    Goes long when RSI enters the oversold region.
    Goes short when RSI enters the overbought region.
    Goes neutral otherwise.
    """
    def __init__(self, rsi_period: int = 14, oversold_threshold: float = 30.0, overbought_threshold: float = 70.0) -> None:
        """
        Initializes the RsiMeanReversionStrategy.

        Args:
            rsi_period: The lookback period for RSI calculation.
            oversold_threshold: The RSI level below which the asset is considered oversold.
            overbought_threshold: The RSI level above which the asset is considered overbought.
        """
        if not isinstance(rsi_period, int) or rsi_period <= 0:
            raise ValueError("RSI period must be a positive integer.")
        if not isinstance(oversold_threshold, (int, float)) or not isinstance(overbought_threshold, (int, float)):
            raise ValueError("RSI thresholds must be numeric.")
        if oversold_threshold >= overbought_threshold:
            raise ValueError("Oversold threshold must be less than overbought threshold.")
        if not (0 < oversold_threshold < 100 and 0 < overbought_threshold < 100):
            raise ValueError("RSI thresholds must be between 0 and 100.")

        super().__init__(rsi_period=rsi_period, oversold_threshold=oversold_threshold, overbought_threshold=overbought_threshold)
        self.rsi_period = rsi_period
        self.oversold = oversold_threshold
        self.overbought = overbought_threshold

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculates the Relative Strength Index (RSI)."""
        delta = series.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Use Exponential Moving Average for smoothing gains and losses
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on RSI levels.

        Args:
            data: DataFrame containing at least the 'Close' price column.

        Returns:
            A pandas Series containing the position signal (1 for long, -1 for short, 0 for neutral).
        """
        if 'Close' not in data.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column.")

        close_prices = data['Close']

        # Calculate RSI
        rsi = self._calculate_rsi(close_prices, self.rsi_period)

        # Determine signal based on RSI thresholds
        position = pd.Series(index=data.index, data=0, dtype=int)
        position[rsi < self.oversold] = 1   # Go long when oversold
        position[rsi > self.overbought] = -1  # Go short when overbought

        # Signals are only valid after RSI calculation period
        position[:self.rsi_period] = 0

        # Fill NaNs that might occur if avg_loss is zero initially
        position.fillna(0, inplace=True)

        return position.astype(int) 