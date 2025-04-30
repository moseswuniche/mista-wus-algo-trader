import pandas as pd
from typing import Tuple, Optional

from .base_strategy import Strategy
from ..technical_indicators import calculate_ema, calculate_sma


class MovingAverageCrossoverStrategy(Strategy):
    """
    A trend-following strategy based on the crossover of two Moving Averages (MAs).
    Can use either Simple Moving Averages (SMA) or Exponential Moving Averages (EMA).
    Goes long when the fast MA crosses above the slow MA.
    Goes short when the fast MA crosses below the slow MA.
    Optionally filters signals based on a long-term trend MA.
    """

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        ma_type: str = "EMA",
        trend_filter_period: Optional[int] = None,
        trend_filter_use_ema: bool = True,
    ) -> None:
        """
        Initializes the MovingAverageCrossoverStrategy.

        Args:
            fast_period: The lookback period for the fast MA.
            slow_period: The lookback period for the slow MA.
            ma_type: The type of moving average to use ("SMA" or "EMA"). Defaults to "EMA".
            trend_filter_period: Optional lookback period for the trend-filtering MA.
            trend_filter_use_ema: If True and trend_filter_period is set, use EMA for trend, else SMA.
        """
        if ma_type not in ["SMA", "EMA"]:
            raise ValueError("ma_type must be either 'SMA' or 'EMA'.")
        if (
            not isinstance(fast_period, int)
            or not isinstance(slow_period, int)
            or fast_period <= 0
            or slow_period <= 0
        ):
            raise ValueError("MA periods must be positive integers.")
        if fast_period >= slow_period:
            raise ValueError("Fast MA period must be less than Slow MA period.")
        if trend_filter_period is not None and (
            not isinstance(trend_filter_period, int) or trend_filter_period <= 0
        ):
            raise ValueError(
                "Trend filter period must be a positive integer if provided."
            )

        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            ma_type=ma_type,
            trend_filter_period=trend_filter_period,
            trend_filter_use_ema=trend_filter_use_ema,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type
        self.trend_filter_period = trend_filter_period
        self.trend_filter_use_ema = trend_filter_use_ema

    def _calculate_ma(self, data: pd.DataFrame, period: int, ma_type: str) -> pd.Series:
        """Calculates the specified Moving Average."""
        if ma_type == "EMA":
            if not all(c in data.columns for c in ["High", "Low", "Close"]):
                raise ValueError("Data needs High, Low, Close for EMA calculation.")
            return calculate_ema(data, period=period)
        elif ma_type == "SMA":
            if "Close" not in data.columns:
                raise ValueError("Data needs Close for SMA.")
            return calculate_sma(data["Close"], period=period)
        else:
            raise ValueError(f"Unknown ma_type: {ma_type}")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on MA crossovers, optionally filtered by trend.

        Args:
            data: DataFrame containing price data required by the chosen MA types.

        Returns:
            A pandas Series containing the position signal (1 for long, -1 for short, 0 for neutral).
        """
        if "Close" not in data.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column.")

        df = data.copy()
        close_prices = df["Close"]

        # Calculate MAs
        fast_ma_col = f"{self.ma_type.lower()}_fast"
        slow_ma_col = f"{self.ma_type.lower()}_slow"
        df[fast_ma_col] = self._calculate_ma(df, self.fast_period, self.ma_type)
        df[slow_ma_col] = self._calculate_ma(df, self.slow_period, self.ma_type)

        # Determine initial signal based on crossover
        df["signal"] = 0
        df.loc[df[fast_ma_col] > df[slow_ma_col], "signal"] = 1  # Go long
        df.loc[df[fast_ma_col] < df[slow_ma_col], "signal"] = -1  # Go short

        # --- Apply Trend Filter ---
        if self.trend_filter_period is not None and self.trend_filter_period > 0:
            trend_ma_type = "EMA" if self.trend_filter_use_ema else "SMA"
            trend_col_name = f"{trend_ma_type.lower()}_trend_{self.trend_filter_period}"
            df[trend_col_name] = self._calculate_ma(
                df, self.trend_filter_period, trend_ma_type
            )

            # Filter signals based on trend
            long_block = (df["signal"] == 1) & (close_prices < df[trend_col_name])
            short_block = (df["signal"] == -1) & (close_prices > df[trend_col_name])
            df.loc[long_block | short_block, "signal"] = 0
        # --- End Trend Filter ---

        # Signals are only valid after MAs and trend filter (if used) have enough data
        warmup_period = self.slow_period
        if self.trend_filter_period:
            warmup_period = max(warmup_period, self.trend_filter_period)

        df.iloc[:warmup_period, df.columns.get_loc("signal")] = 0

        # Fill NaNs and ensure integer type
        df["signal"] = df["signal"].fillna(0)
        return df["signal"].astype(int)
