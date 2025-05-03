import pandas as pd
from typing import Tuple, Optional

from .base_strategy import Strategy
from ..technical_indicators import calculate_ema


class BollingerBandReversionStrategy(Strategy):
    """
    A mean-reversion strategy using Bollinger Bands.
    Goes long when the price touches or crosses below the lower band.
    Goes short when the price touches or crosses above the upper band.
    Optionally filters signals based on a long-term trend EMA.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        trend_filter_period: Optional[int] = None,
        trend_filter_use_ema: bool = True,
    ) -> None:
        """
        Initializes the BollingerBandReversionStrategy.

        Args:
            bb_period: The lookback period for the moving average and standard deviation.
            bb_std_dev: The number of standard deviations for the upper and lower bands.
            trend_filter_period: Optional lookback period for the trend-filtering MA.
            trend_filter_use_ema: If True and trend_filter_period is set, use EMA, else SMA.
        """
        if not isinstance(bb_period, int) or bb_period <= 0:
            raise ValueError("Bollinger Band period must be a positive integer.")
        if not isinstance(bb_std_dev, (int, float)) or bb_std_dev <= 0:
            raise ValueError(
                "Bollinger Band standard deviation must be a positive number."
            )
        if trend_filter_period is not None and (
            not isinstance(trend_filter_period, int) or trend_filter_period <= 0
        ):
            raise ValueError(
                "Trend filter period must be a positive integer if provided."
            )

        super().__init__(
            bb_period=bb_period,
            bb_std_dev=bb_std_dev,
            trend_filter_period=trend_filter_period,
            trend_filter_use_ema=trend_filter_use_ema,
        )
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.trend_filter_period = trend_filter_period
        self.trend_filter_use_ema = trend_filter_use_ema

    def _calculate_bollinger_bands(
        self, series: pd.Series, period: int, std_dev: float
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculates Bollinger Bands (Middle, Upper, Lower)."""
        middle_band = series.rolling(window=period).mean()
        rolling_std = series.rolling(window=period).std()
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        return middle_band, upper_band, lower_band

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals based on Bollinger Band levels, optionally filtered by trend.

        Args:
            data: DataFrame containing at least the 'close' price column (lowercase).
                  Requires 'high', 'low', 'close' if trend filter is EMA.

        Returns:
            A pandas DataFrame containing the position signal (1 for long, -1 for short, 0 for neutral).
        """
        required_close_col = "close"
        if required_close_col not in data.columns:
            raise ValueError(
                f"Input DataFrame must contain a '{required_close_col}' column."
            )

        required_hlc_cols = ["high", "low", "close"]
        if (
            self.trend_filter_period
            and self.trend_filter_use_ema
            and not all(c in data.columns for c in required_hlc_cols)
        ):
            raise ValueError(
                f"Data must contain {required_hlc_cols} for EMA trend filter."
            )

        df = data.copy()
        close_prices = df[required_close_col]

        # Calculate Bollinger Bands
        middle, upper, lower = self._calculate_bollinger_bands(
            close_prices, self.bb_period, self.bb_std_dev
        )
        df["bb_middle"] = middle
        df["bb_upper"] = upper
        df["bb_lower"] = lower

        # Determine initial signal based on band touches/crosses
        df["signal"] = 0
        df.loc[close_prices <= df["bb_lower"], "signal"] = 1  # Touch or cross below
        df.loc[close_prices >= df["bb_upper"], "signal"] = -1  # Touch or cross above

        # --- Apply Trend Filter ---
        if self.trend_filter_period is not None and self.trend_filter_period > 0:
            trend_col_name = f"trend_ma_{self.trend_filter_period}"
            if self.trend_filter_use_ema:
                df[trend_col_name] = calculate_ema(df, period=self.trend_filter_period)
            else:
                df[trend_col_name] = (
                    df[required_close_col]
                    .rolling(window=self.trend_filter_period)
                    .mean()
                )

            # Filter signals based on trend
            # Block longs in downtrend
            long_block_condition = (df["signal"] == 1) & (
                close_prices < df[trend_col_name]
            )
            df.loc[long_block_condition, "signal"] = 0

            # Block shorts in uptrend
            short_block_condition = (df["signal"] == -1) & (
                close_prices > df[trend_col_name]
            )
            df.loc[short_block_condition, "signal"] = 0
        # --- End Trend Filter ---

        # Signals are only valid after the initial BB period AND trend period (if used)
        if self.trend_filter_period is not None and self.trend_filter_period > 0:
            warmup_period = max(self.bb_period, self.trend_filter_period)
        else:
            warmup_period = self.bb_period

        # Use .iloc to set initial signals to avoid SettingWithCopyWarning
        df.iloc[:warmup_period, df.columns.get_loc("signal")] = 0

        # Fill NaNs (e.g., from initial MA/BB calculation)
        df["signal"] = df["signal"].fillna(0)

        # Return the entire DataFrame with the signal column
        return df
