import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

from .base_strategy import Strategy


class LongShortStrategy(Strategy):
    """
    A strategy that goes long/short based on return Z-scores,
    filtered by volume change Z-scores (defined by volume_thresh) and optionally by long-term trend EMA.
    Note: Dynamic beta hedging for position sizing is suggested but not implemented here; handle in portfolio/execution layer.
    """

    def __init__(
        self,
        volume_thresh: Tuple[float, float],
        return_z_score_period: int = 20, # Lookback period for return Z-score calculation
        return_z_score_threshold: float = 2.0, # Z-score threshold (e.g., 2.0 means +/- 2 standard deviations)
        trend_filter_period: Optional[int] = None,
    ) -> None:
        """
        Initializes the LongShortStrategy.

        Args:
            volume_thresh: Tuple containing the lower and upper log volume change thresholds (can represent Z-scores if volume change is standardized).
            return_z_score_period: Lookback period for calculating return Z-score.
            return_z_score_threshold: The absolute Z-score value to trigger signals.
            trend_filter_period: Optional lookback period for the trend-filtering EMA. If None or 0, no filter is applied.
        """
        # Validate thresholds (basic check)
        if not (
            isinstance(volume_thresh, tuple)
            and len(volume_thresh) == 2
            and volume_thresh[0] < volume_thresh[1]
        ):
            raise ValueError(
                "volume_thresh must be a tuple of two floats (low, high) with low < high."
            )
        if not isinstance(return_z_score_period, int) or return_z_score_period <= 1:
            raise ValueError("Return Z-score period must be an integer greater than 1.")
        if not isinstance(return_z_score_threshold, (int, float)) or return_z_score_threshold <= 0:
            raise ValueError("Return Z-score threshold must be a positive number.")
        if trend_filter_period is not None and (
            not isinstance(trend_filter_period, int) or trend_filter_period <= 0
        ):
            raise ValueError(
                "Trend filter period must be a positive integer if provided."
            )

        super().__init__(
            volume_thresh=volume_thresh,
            return_z_score_period=return_z_score_period,
            return_z_score_threshold=return_z_score_threshold,
            trend_filter_period=trend_filter_period,
        )
        self.volume_thresh = volume_thresh
        self.z_period = return_z_score_period
        self.z_thresh = return_z_score_threshold
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

        # Calculate volume change (log ratio), handling potential zero volume
        volume_ratio = df.Volume.div(df.Volume.shift(1))
        volume_ratio.loc[volume_ratio <= 0] = np.nan
        df["vol_ch"] = np.log(volume_ratio)
        # NOTE: If volume_thresh represents Z-scores, vol_ch should be standardized here.
        # Example: df['vol_ch_z'] = (df['vol_ch'] - df['vol_ch'].rolling(window=Z_VOL_PERIOD).mean()) / df['vol_ch'].rolling(window=Z_VOL_PERIOD).std()
        # Then use df['vol_ch_z'] in the condition below.

        # Calculate Z-score of returns
        rolling_mean_ret = df["returns"].rolling(window=self.z_period).mean()
        rolling_std_ret = df["returns"].rolling(window=self.z_period).std()
        df["return_z"] = (df["returns"] - rolling_mean_ret) / rolling_std_ret.replace(0, np.nan)

        # --- Calculate Initial Signal ---
        cond_long_ret = df["return_z"] <= -self.z_thresh # Long on negative Z-score
        cond_short_ret = df["return_z"] >= self.z_thresh # Short on positive Z-score
        cond_vol = df.vol_ch.between(self.volume_thresh[0], self.volume_thresh[1]) # Use vol_ch_z if standardized

        df["signal"] = 0
        df.loc[cond_long_ret & cond_vol, "signal"] = 1
        df.loc[cond_short_ret & cond_vol, "signal"] = -1

        # --- Apply Trend Filter ---
        if self.trend_filter_period is not None and self.trend_filter_period > 0:
            # Calculate trend EMA
            ema_trend_col = f"ema_{self.trend_filter_period}"
            df[ema_trend_col] = close_prices.ewm(
                span=self.trend_filter_period, adjust=False, min_periods=self.trend_filter_period
            ).mean()

            # Filter signals based on trend
            # Block longs in downtrend
            long_block_condition = (df["signal"] == 1) & (close_prices < df[ema_trend_col])
            df.loc[long_block_condition, "signal"] = 0

            # Block shorts in uptrend
            short_block_condition = (df["signal"] == -1) & (close_prices > df[ema_trend_col])
            df.loc[short_block_condition, "signal"] = 0
        # --- End Trend Filter ---

        # Determine warmup based on calculations needed (returns/vol need 1, Z-score needs z_period, EMA needs trend_filter_period)
        warmup_period = max(1, self.z_period)
        if self.trend_filter_period is not None and self.trend_filter_period > 0:
            warmup_period = max(warmup_period, self.trend_filter_period)

        # Use .loc to set initial signals to avoid SettingWithCopyWarning
        df.loc[df.index[:warmup_period], "signal"] = 0

        # Fill NaNs (from initial shifts or SMA calculation)
        # Assign back instead of using inplace=True to avoid warnings
        df["signal"] = df["signal"].fillna(0)

        return df["signal"].astype(int)
