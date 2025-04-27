import pandas as pd
from typing import Tuple

from .base_strategy import Strategy

class BollingerBandReversionStrategy(Strategy):
    """
    A mean-reversion strategy using Bollinger Bands.
    Goes long when the price touches or crosses below the lower band.
    Goes short when the price touches or crosses above the upper band.
    Goes neutral otherwise.
    """
    def __init__(self, bb_period: int = 20, bb_std_dev: float = 2.0) -> None:
        """
        Initializes the BollingerBandReversionStrategy.

        Args:
            bb_period: The lookback period for the moving average and standard deviation.
            bb_std_dev: The number of standard deviations for the upper and lower bands.
        """
        if not isinstance(bb_period, int) or bb_period <= 0:
            raise ValueError("Bollinger Band period must be a positive integer.")
        if not isinstance(bb_std_dev, (int, float)) or bb_std_dev <= 0:
            raise ValueError("Bollinger Band standard deviation must be a positive number.")

        super().__init__(bb_period=bb_period, bb_std_dev=bb_std_dev)
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev

    def _calculate_bollinger_bands(self, series: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculates Bollinger Bands (Middle, Upper, Lower)."""
        middle_band = series.rolling(window=period).mean()
        rolling_std = series.rolling(window=period).std()
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        return middle_band, upper_band, lower_band

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on Bollinger Band levels.

        Args:
            data: DataFrame containing at least the 'Close' price column.

        Returns:
            A pandas Series containing the position signal (1 for long, -1 for short, 0 for neutral).
        """
        if 'Close' not in data.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column.")

        close_prices = data['Close']

        # Calculate Bollinger Bands
        middle, upper, lower = self._calculate_bollinger_bands(close_prices, self.bb_period, self.bb_std_dev)

        # Determine signal based on band touches/crosses
        position = pd.Series(index=data.index, data=0, dtype=int)
        position[close_prices < lower] = 1   # Go long when below lower band
        position[close_prices > upper] = -1  # Go short when above upper band

        # Signals are only valid after the initial BB period
        position[:self.bb_period] = 0

        # Fill NaNs
        position.fillna(0, inplace=True)

        return position.astype(int) 