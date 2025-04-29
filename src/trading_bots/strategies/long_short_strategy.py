import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

from .base_strategy import Strategy


class LongShortStrategy(Strategy):
    """
    A strategy that goes long on negative returns and short on positive returns,
    filtered by volume change and optionally by long-term trend.
    """

    def __init__(
        self,
        return_thresh: Tuple[float, float],
        volume_thresh: Tuple[float, float],
        trend_filter_period: Optional[int] = None,
    ) -> None:
        """
        Initializes the LongShortStrategy.

        Args:
            return_thresh: Tuple containing the lower and upper log return thresholds.
            volume_thresh: Tuple containing the lower and upper log volume change thresholds.
            trend_filter_period: Optional lookback period for the trend-filtering SMA. If None or 0, no filter is applied.
        """
        # Validate thresholds (basic check)
        if not (
            isinstance(return_thresh, tuple)
            and len(return_thresh) == 2
            and return_thresh[0] < return_thresh[1]
        ):
            raise ValueError(
                "return_thresh must be a tuple of two floats (low, high) with low < high."
            )
        if not (
            isinstance(volume_thresh, tuple)
            and len(volume_thresh) == 2
            and volume_thresh[0] < volume_thresh[1]
        ):
            raise ValueError(
                "volume_thresh must be a tuple of two floats (low, high) with low < high."
            )
        if trend_filter_period is not None and (
            not isinstance(trend_filter_period, int) or trend_filter_period <= 0
        ):
            raise ValueError(
                "Trend filter period must be a positive integer if provided."
            )

        super().__init__(
            return_thresh=return_thresh,
            volume_thresh=volume_thresh,
            trend_filter_period=trend_filter_period,
        )
        self.return_thresh = return_thresh
        self.volume_thresh = volume_thresh
        self.trend_filter_period = trend_filter_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on return, volume change, and optional trend filter.

        Args:
            data: DataFrame containing 'Close' and 'Volume' columns.

        Returns:
            A pandas Series containing the position signal (1, -1, or 0).
        """
        if not all(col in data.columns for col in ["Close", "Volume"]):
            raise ValueError(
                "Input DataFrame must contain 'Close' and 'Volume' columns."
            )

        df = data[["Close", "Volume"]].copy()
        close_prices = df["Close"]

        df["returns"] = np.log(df.Close / df.Close.shift())

        # Calculate volume change, handling potential zero volume
        volume_ratio = df.Volume.div(df.Volume.shift(1))
        volume_ratio.loc[volume_ratio <= 0] = np.nan
        df["vol_ch"] = np.log(volume_ratio)

        # --- Calculate Initial Signal ---
        cond_long_ret = df.returns <= self.return_thresh[0]
        cond_short_ret = df.returns >= self.return_thresh[1]
        cond_vol = df.vol_ch.between(self.volume_thresh[0], self.volume_thresh[1])

        df["signal"] = 0
        df.loc[cond_long_ret & cond_vol, "signal"] = 1
        df.loc[cond_short_ret & cond_vol, "signal"] = -1

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

        # Determine warmup based on calculations needed (returns/vol need 1, SMA needs trend_filter_period)
        warmup_period = 1
        if self.trend_filter_period is not None and self.trend_filter_period > 0:
            warmup_period = max(warmup_period, self.trend_filter_period)

        # Use .loc to set initial signals to avoid SettingWithCopyWarning
        df.loc[df.index[:warmup_period], "signal"] = 0

        # Fill NaNs (from initial shifts or SMA calculation)
        # Assign back instead of using inplace=True to avoid warnings
        df["signal"] = df["signal"].fillna(0)

        return df["signal"].astype(int)
