import pandas as pd
from typing import Tuple, Optional

from .base_strategy import Strategy


class MovingAverageCrossoverStrategy(Strategy):
    """
    A trend-following strategy based on the crossover of two Exponential Moving Averages (EMAs).
    Goes long when the fast EMA crosses above the slow EMA.
    Goes short when the fast EMA crosses below the slow EMA.
    Optionally filters signals based on a long-term trend SMA.
    """

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        trend_filter_period: Optional[int] = None,
    ) -> None:
        """
        Initializes the MovingAverageCrossoverStrategy.

        Args:
            fast_period: The lookback period for the fast EMA.
            slow_period: The lookback period for the slow EMA.
            trend_filter_period: Optional lookback period for the trend-filtering SMA. If None or 0, no filter is applied.
        """
        if (
            not isinstance(fast_period, int)
            or not isinstance(slow_period, int)
            or fast_period <= 0
            or slow_period <= 0
        ):
            raise ValueError("EMA periods must be positive integers.")
        if fast_period >= slow_period:
            raise ValueError("Fast EMA period must be less than Slow EMA period.")
        if trend_filter_period is not None and (
            not isinstance(trend_filter_period, int) or trend_filter_period <= 0
        ):
            raise ValueError(
                "Trend filter period must be a positive integer if provided."
            )

        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            trend_filter_period=trend_filter_period,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trend_filter_period = trend_filter_period

    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculates Exponential Moving Average."""
        # Using adjust=False aligns better with many trading platforms' EMA calculations
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on EMA crossovers, optionally filtered by trend.

        Args:
            data: DataFrame containing at least the 'Close' price column.

        Returns:
            A pandas Series containing the position signal (1 for long, -1 for short, 0 for neutral).
        """
        if "Close" not in data.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column.")

        df = data.copy()
        close_prices = df["Close"]

        # Calculate EMAs
        df["ema_fast"] = self._calculate_ema(close_prices, self.fast_period)
        df["ema_slow"] = self._calculate_ema(close_prices, self.slow_period)

        # Determine initial signal based on crossover
        df["signal"] = 0
        df.loc[df["ema_fast"] > df["ema_slow"], "signal"] = 1  # Go long
        df.loc[df["ema_fast"] < df["ema_slow"], "signal"] = -1  # Go short

        # --- Apply Trend Filter ---
        if self.trend_filter_period is not None and self.trend_filter_period > 0:
            # Calculate trend SMA
            sma_trend_col = f"sma_{self.trend_filter_period}"
            df[sma_trend_col] = close_prices.rolling(
                window=self.trend_filter_period, min_periods=self.trend_filter_period
            ).mean()

            # Filter signals based on trend
            # Block longs in downtrend
            long_block_condition = (df["signal"] == 1) & (
                close_prices < df[sma_trend_col]
            )
            df.loc[long_block_condition, "signal"] = 0

            # Block shorts in uptrend
            short_block_condition = (df["signal"] == -1) & (
                close_prices > df[sma_trend_col]
            )
            df.loc[short_block_condition, "signal"] = 0
        # --- End Trend Filter ---

        # Signals are only valid after EMAs and trend filter (if used) have enough data
        warmup_period = max(self.fast_period, self.slow_period)
        if self.trend_filter_period is not None and self.trend_filter_period > 0:
            warmup_period = max(warmup_period, self.trend_filter_period)

        # Use .loc to set initial signals to avoid SettingWithCopyWarning
        df.loc[df.index[:warmup_period], "signal"] = 0

        # Fill any remaining NaNs (e.g., from initial SMA calculation)
        # Assign back instead of using inplace=True to avoid warnings
        df["signal"] = df["signal"].fillna(0)

        return df["signal"].astype(int)
