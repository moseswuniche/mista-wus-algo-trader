import pandas as pd
from typing import Tuple, Optional
import numpy as np

from .base_strategy import Strategy
from trading_bots.helpers.technical_indicators import calculate_atr # Assuming ATR calculation helper exists


class RsiMeanReversionStrategy(Strategy):
    """
    A mean-reversion strategy based on the Relative Strength Index (RSI).
    Goes long when RSI enters the oversold region.
    Goes short when RSI enters the overbought region.
    Optionally filters signals based on a long-term trend EMA.
    Optionally uses ATR-adjusted thresholds for dynamic entry points.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        trend_filter_period: Optional[int] = None,
        atr_period: Optional[int] = 14, # Period for ATR calculation
        atr_threshold_multiplier: Optional[float] = None, # Multiplier for ATR adjustment, e.g., 0.5 or 1.0. If None, fixed thresholds are used.
    ) -> None:
        """
        Initializes the RsiMeanReversionStrategy.

        Args:
            rsi_period: The lookback period for RSI calculation.
            oversold_threshold: The RSI level below which the asset is considered oversold (base level if ATR adjusted).
            overbought_threshold: The RSI level above which the asset is considered overbought (base level if ATR adjusted).
            trend_filter_period: Optional lookback period for the trend-filtering EMA. If None or 0, no filter is applied.
            atr_period: Optional lookback period for ATR calculation for dynamic thresholds. Defaults to 14.
            atr_threshold_multiplier: Optional multiplier for ATR threshold adjustment. If None, fixed thresholds are used.
        """
        if not isinstance(rsi_period, int) or rsi_period <= 0:
            raise ValueError("RSI period must be a positive integer.")
        if not isinstance(oversold_threshold, (int, float)) or not isinstance(
            overbought_threshold, (int, float)
        ):
            raise ValueError("RSI thresholds must be numeric.")
        if oversold_threshold >= overbought_threshold:
            raise ValueError(
                "Oversold threshold must be less than overbought threshold."
            )
        if not (0 < oversold_threshold < 100 and 0 < overbought_threshold < 100):
            raise ValueError("RSI thresholds must be between 0 and 100.")
        if trend_filter_period is not None and (
            not isinstance(trend_filter_period, int) or trend_filter_period <= 0
        ):
            raise ValueError(
                "Trend filter period must be a positive integer if provided."
            )
        if atr_period is not None and (not isinstance(atr_period, int) or atr_period <= 0):
            raise ValueError("ATR period must be a positive integer if provided.")
        if atr_threshold_multiplier is not None and not isinstance(atr_threshold_multiplier, (int, float)):
            raise ValueError("ATR threshold multiplier must be numeric if provided.")

        super().__init__(
            rsi_period=rsi_period,
            oversold_threshold=oversold_threshold,
            overbought_threshold=overbought_threshold,
            trend_filter_period=trend_filter_period,
            atr_period=atr_period,
            atr_threshold_multiplier=atr_threshold_multiplier,
        )
        self.rsi_period = rsi_period
        self.oversold = oversold_threshold
        self.overbought = overbought_threshold
        self.trend_filter_period = trend_filter_period
        self.use_atr_thresholds = atr_threshold_multiplier is not None and atr_period is not None
        self.atr_period = atr_period
        self.atr_multiplier = atr_threshold_multiplier

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculates the Relative Strength Index (RSI)."""
        delta = series.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Use Exponential Moving Average for smoothing gains and losses
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        # Handle potential division by zero if avg_loss is 0
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi.fillna(50, inplace=True)

        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on RSI levels, optionally filtered by trend.

        Args:
            data: DataFrame containing at least the 'Close' price column.

        Returns:
            A pandas Series containing the position signal (1 for long, -1 for short, 0 for neutral).
        """
        if "Close" not in data.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column.")

        # Required columns for ATR calculation
        if self.use_atr_thresholds and not all(col in data.columns for col in ["High", "Low", "Close"]):
             raise ValueError("Input DataFrame must contain 'High', 'Low', 'Close' columns for ATR calculation.")

        df = data.copy()
        close_prices = df["Close"]

        # Calculate RSI
        df["rsi"] = self._calculate_rsi(close_prices, self.rsi_period)

        # Determine initial signal based on RSI thresholds
        oversold_level = self.oversold
        overbought_level = self.overbought

        # --- ATR Dynamic Threshold Adjustment ---
        if self.use_atr_thresholds:
            df["atr"] = calculate_atr(df, period=self.atr_period) # Assume helper exists
            # Adjust thresholds: Lower overbought, raise oversold based on ATR.
            # Use a normalized ATR or scale appropriately if RSI and ATR scales differ significantly.
            # For simplicity, directly using ATR * multiplier. Consider normalization if needed.
            # Ensure thresholds stay within logical bounds (e.g., > 0, < 100)
            overbought_level = (self.overbought - df["atr"] * self.atr_multiplier).clip(lower=self.oversold + 1) # Ensure OB > OS
            oversold_level = (self.oversold + df["atr"] * self.atr_multiplier).clip(upper=self.overbought - 1) # Ensure OS < OB

        df["signal"] = 0
        # Note: Using dynamic levels potentially requires Series comparison
        df.loc[df["rsi"] < oversold_level, "signal"] = 1  # Go long when oversold
        df.loc[df["rsi"] > overbought_level, "signal"] = -1  # Go short when overbought

        # --- Apply Trend Filter ---
        if self.trend_filter_period is not None and self.trend_filter_period > 0:
            # Calculate trend EMA
            ema_trend_col = f"ema_{self.trend_filter_period}"
            df[ema_trend_col] = close_prices.ewm(
                span=self.trend_filter_period, adjust=False, min_periods=self.trend_filter_period
            ).mean()

            # Filter signals based on trend
            # Block longs in downtrend
            long_block_condition = (df["signal"] == 1) & (
                close_prices < df[ema_trend_col]
            )
            df.loc[long_block_condition, "signal"] = 0

            # Block shorts in uptrend
            short_block_condition = (df["signal"] == -1) & (
                close_prices > df[ema_trend_col]
            )
            df.loc[short_block_condition, "signal"] = 0
        # --- End Trend Filter ---

        # Signals are only valid after RSI and trend filter (if used) have enough data
        warmup_period = self.rsi_period
        if self.trend_filter_period is not None and self.trend_filter_period > 0:
            warmup_period = max(warmup_period, self.trend_filter_period)
        if self.use_atr_thresholds:
            warmup_period = max(warmup_period, self.atr_period)

        # Use .loc to set initial signals to avoid SettingWithCopyWarning
        df.loc[df.index[:warmup_period], "signal"] = 0

        # Fill NaNs that might occur if avg_loss is zero initially or from SMA
        # Assign back instead of using inplace=True to avoid warnings
        df["signal"] = df["signal"].fillna(0)

        return df["signal"].astype(int)
