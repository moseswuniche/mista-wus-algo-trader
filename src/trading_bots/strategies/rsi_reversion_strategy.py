import pandas as pd
from typing import Tuple, Optional
import numpy as np

from .base_strategy import Strategy
from ..technical_indicators import calculate_atr, calculate_ema


class RsiMeanReversionStrategy(Strategy):
    """
    A mean-reversion strategy based on the Relative Strength Index (RSI).
    Goes long when RSI enters the oversold region.
    Goes short when RSI enters the overbought region.
    Optionally filters signals based on a long-term trend MA.
    Optionally uses ATR-adjusted thresholds for dynamic entry points.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        trend_filter_period: Optional[int] = None,
        trend_filter_use_ema: bool = True,
        atr_period: Optional[int] = 14,  # Period for ATR calculation
        atr_threshold_multiplier: Optional[
            float
        ] = None,  # Multiplier for ATR adjustment, e.g., 0.5 or 1.0. If None, fixed thresholds are used.
    ) -> None:
        """
        Initializes the RsiMeanReversionStrategy.

        Args:
            rsi_period: The lookback period for RSI calculation.
            oversold_threshold: The RSI level below which the asset is considered oversold (base level if ATR adjusted).
            overbought_threshold: The RSI level above which the asset is considered overbought (base level if ATR adjusted).
            trend_filter_period: Optional lookback period for the trend-filtering MA.
            trend_filter_use_ema: If True and trend_filter_period is set, use EMA, else SMA.
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
        if atr_period is not None and (
            not isinstance(atr_period, int) or atr_period <= 0
        ):
            raise ValueError("ATR period must be a positive integer if provided.")
        if atr_threshold_multiplier is not None and not isinstance(
            atr_threshold_multiplier, (int, float)
        ):
            raise ValueError("ATR threshold multiplier must be numeric if provided.")

        super().__init__(
            rsi_period=rsi_period,
            oversold_threshold=oversold_threshold,
            overbought_threshold=overbought_threshold,
            trend_filter_period=trend_filter_period,
            trend_filter_use_ema=trend_filter_use_ema,
            atr_period=atr_period,
            atr_threshold_multiplier=atr_threshold_multiplier,
        )
        self.rsi_period = rsi_period
        self.oversold_base = oversold_threshold
        self.overbought_base = overbought_threshold
        self.trend_filter_period = trend_filter_period
        self.trend_filter_use_ema = trend_filter_use_ema
        self.use_atr_thresholds = (
            atr_threshold_multiplier is not None and atr_period is not None
        )
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
        Generates trading signals based on RSI levels, optionally filtered by trend
        and using dynamic ATR thresholds.

        Args:
            data: DataFrame containing at least 'Close'.
                  Requires 'High', 'Low', 'Close' if ATR thresholds or EMA trend filter are used.
                  Expects 'atr' column pre-calculated if self.use_atr_thresholds is True.

        Returns:
            A pandas Series containing the position signal (1 for long, -1 for short, 0 for neutral).
        """
        required_cols = ["Close"]
        if self.use_atr_thresholds:
            required_cols.extend(["High", "Low", "Close", "atr"])
        if self.trend_filter_period and self.trend_filter_use_ema:
            required_cols.extend(["High", "Low", "Close"])
        required_cols = list(set(required_cols))  # Remove duplicates
        if not all(col in data.columns for col in required_cols):
            missing = [c for c in required_cols if c not in data.columns]
            raise ValueError(f"Input DataFrame missing required columns: {missing}")

        df = data.copy()
        close_prices = df["Close"]

        # Calculate RSI
        df["rsi"] = self._calculate_rsi(close_prices, self.rsi_period)

        # Determine dynamic or fixed thresholds
        oversold_level = self.oversold_base
        overbought_level = self.overbought_base

        if self.use_atr_thresholds:
            # Use pre-calculated ATR (ensure atr_multiplier is float for calculation)
            atr_multiplier = (
                float(self.atr_multiplier) if self.atr_multiplier is not None else 0.0
            )
            # Adjust thresholds: Lower overbought, raise oversold based on ATR.
            # Note: Scaling might be needed if RSI and ATR scales differ significantly.
            # Using a simplified direct adjustment here.
            overbought_level = (self.overbought_base - df["atr"] * atr_multiplier).clip(
                lower=self.oversold_base + 1
            )
            oversold_level = (self.oversold_base + df["atr"] * atr_multiplier).clip(
                upper=self.overbought_base - 1
            )
            df["oversold_dynamic"] = (
                oversold_level  # Store for debugging/analysis if needed
            )
            df["overbought_dynamic"] = overbought_level

        # Determine initial signal based on RSI crossing thresholds
        df["signal"] = 0
        df.loc[df["rsi"] <= oversold_level, "signal"] = (
            1  # Enter long on touch/cross below oversold
        )
        df.loc[df["rsi"] >= overbought_level, "signal"] = (
            -1
        )  # Enter short on touch/cross above overbought

        # --- Apply Trend Filter ---
        if self.trend_filter_period is not None and self.trend_filter_period > 0:
            trend_col_name = f"trend_ma_{self.trend_filter_period}"
            if self.trend_filter_use_ema:
                df[trend_col_name] = calculate_ema(df, period=self.trend_filter_period)
            else:
                df[trend_col_name] = close_prices.rolling(
                    window=self.trend_filter_period
                ).mean()

            # Filter signals based on trend
            long_block = (df["signal"] == 1) & (close_prices < df[trend_col_name])
            short_block = (df["signal"] == -1) & (close_prices > df[trend_col_name])
            df.loc[long_block | short_block, "signal"] = 0
        # --- End Trend Filter ---

        # Determine warmup period
        warmup_period = self.rsi_period
        if self.trend_filter_period:
            warmup_period = max(warmup_period, self.trend_filter_period)
        if self.use_atr_thresholds and self.atr_period:
            warmup_period = max(warmup_period, self.atr_period)

        # Set initial signals to 0 during warmup
        df.iloc[:warmup_period, df.columns.get_loc("signal")] = 0

        # Fill NaNs and ensure integer type
        df["signal"] = df["signal"].fillna(0)
        return df["signal"].astype(int)
