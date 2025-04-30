import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

from .base_strategy import Strategy

# Import necessary indicator functions
from ..technical_indicators import calculate_sma, calculate_ema


class LongShortStrategy(Strategy):
    """
    A strategy that goes long/short based on return Z-scores,
    filtered by volume change Z-scores and optionally by a long-term trend MA (SMA or EMA).
    Note: Dynamic beta hedging for position sizing is suggested but not implemented here; handle in portfolio/execution layer.
    """

    def __init__(
        self,
        return_z_period: int = 20,  # Renamed from return_z_score_period
        return_z_threshold: float = 2.0,  # Renamed from return_z_score_threshold
        volume_z_period: int = 20,
        volume_z_threshold: Tuple[float, float] = (
            -2.0,
            2.0,
        ),  # Expects (low_z, high_z)
        trend_filter_period: Optional[int] = None,
        trend_filter_use_ema: bool = True,  # Added for consistency
    ) -> None:
        """
        Initializes the LongShortStrategy.

        Args:
            return_z_period: Lookback period for calculating return Z-score.
            return_z_threshold: The absolute Z-score value for returns to trigger signals.
            volume_z_period: Lookback period for calculating volume change Z-score.
            volume_z_threshold: Tuple (low, high) containing the Z-score thresholds for volume change filtering.
            trend_filter_period: Optional lookback period for the trend-filtering MA. If None or 0, no filter is applied.
            trend_filter_use_ema: If True and trend_filter_period is set, use EMA, else SMA.
        """
        # Validate thresholds and periods
        if not isinstance(return_z_period, int) or return_z_period <= 1:
            raise ValueError("Return Z-score period must be an integer greater than 1.")
        if not isinstance(return_z_threshold, (int, float)) or return_z_threshold <= 0:
            raise ValueError("Return Z-score threshold must be a positive number.")
        if not isinstance(volume_z_period, int) or volume_z_period <= 1:
            raise ValueError("Volume Z-score period must be an integer greater than 1.")
        if not (
            isinstance(volume_z_threshold, (tuple, list))
            and len(volume_z_threshold) == 2
            # Allow Z-scores to be equal, e.g. (-2, 2) or (2, -2) depending on interpretation
            # and isinstance(volume_z_threshold[0], (int, float))
            # and isinstance(volume_z_threshold[1], (int, float))
            # Logic will handle low/high appropriately
        ):
            raise ValueError(
                "volume_z_threshold must be a tuple or list of two numbers (low Z, high Z)."
            )
        if trend_filter_period is not None and (
            not isinstance(trend_filter_period, int) or trend_filter_period <= 0
        ):
            raise ValueError(
                "Trend filter period must be a positive integer if provided."
            )

        super().__init__(
            return_z_period=return_z_period,
            return_z_threshold=return_z_threshold,
            volume_z_period=volume_z_period,
            volume_z_threshold=volume_z_threshold,
            trend_filter_period=trend_filter_period,
            trend_filter_use_ema=trend_filter_use_ema,
        )
        # Store parameters with corrected names
        self.return_z_period = return_z_period
        self.return_z_thresh = return_z_threshold
        self.volume_z_period = volume_z_period
        self.volume_z_thresh_low = min(volume_z_threshold)  # Ensure low is min
        self.volume_z_thresh_high = max(volume_z_threshold)  # Ensure high is max
        self.trend_filter_period = trend_filter_period
        self.trend_filter_use_ema = trend_filter_use_ema

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on return Z-scores, volume Z-scores, and optional trend filter.

        Args:
            data: DataFrame containing 'Close' and 'Volume' columns. Needs 'High', 'Low' if trend filter uses EMA.

        Returns:
            A pandas Series containing the position signal (1, -1, or 0).
        """
        required_cols = ["Close", "Volume"]
        if self.trend_filter_period and self.trend_filter_use_ema:
            required_cols.extend(["High", "Low"])
        required_cols = list(set(required_cols))
        if not all(col in data.columns for col in required_cols):
            missing = [c for c in required_cols if c not in data.columns]
            raise ValueError(f"Input DataFrame missing required columns: {missing}")

        df = data[required_cols].copy()
        close_prices = df["Close"]

        # Calculate log returns
        df["returns"] = np.log(df.Close / df.Close.shift())

        # Calculate log volume change
        volume_ratio = df.Volume.div(df.Volume.shift(1))
        volume_ratio.loc[volume_ratio <= 0] = np.nan  # Handle zero volume
        df["vol_ch"] = np.log(volume_ratio)

        # Calculate Z-score of log volume change
        rolling_mean_vol = df["vol_ch"].rolling(window=self.volume_z_period).mean()
        rolling_std_vol = df["vol_ch"].rolling(window=self.volume_z_period).std()
        df["volume_z"] = (df["vol_ch"] - rolling_mean_vol) / rolling_std_vol.replace(
            0, np.nan
        )

        # Calculate Z-score of log returns
        rolling_mean_ret = df["returns"].rolling(window=self.return_z_period).mean()
        rolling_std_ret = df["returns"].rolling(window=self.return_z_period).std()
        df["return_z"] = (df["returns"] - rolling_mean_ret) / rolling_std_ret.replace(
            0, np.nan
        )

        # --- Calculate Initial Signal based on Z-scores ---
        # Long condition: Return Z-score is low AND Volume Z-score is low
        cond_long = (df["return_z"] <= -self.return_z_thresh) & (
            df["volume_z"] <= self.volume_z_thresh_low
        )
        # Short condition: Return Z-score is high AND Volume Z-score is high
        cond_short = (df["return_z"] >= self.return_z_thresh) & (
            df["volume_z"] >= self.volume_z_thresh_high
        )

        df["signal"] = 0
        df.loc[cond_long, "signal"] = 1
        df.loc[cond_short, "signal"] = -1

        # --- Apply Trend Filter ---
        if self.trend_filter_period is not None and self.trend_filter_period > 0:
            trend_col_name = f"trend_ma_{self.trend_filter_period}"
            if self.trend_filter_use_ema:
                df[trend_col_name] = calculate_ema(df, period=self.trend_filter_period)
            else:
                df[trend_col_name] = calculate_sma(
                    close_prices, period=self.trend_filter_period
                )

            # Filter signals based on trend
            long_block = (df["signal"] == 1) & (close_prices < df[trend_col_name])
            short_block = (df["signal"] == -1) & (close_prices > df[trend_col_name])
            df.loc[long_block | short_block, "signal"] = 0
        # --- End Trend Filter ---

        # Determine warmup period based on max lookback needed
        warmup_period = max(1, self.return_z_period, self.volume_z_period)
        if self.trend_filter_period is not None and self.trend_filter_period > 0:
            warmup_period = max(warmup_period, self.trend_filter_period)

        # Set initial signals to 0 during warmup
        df.iloc[:warmup_period, df.columns.get_loc("signal")] = 0

        # Fill NaNs and ensure integer type
        df["signal"] = df["signal"].fillna(0)

        return df["signal"].astype(int)
