import pandas as pd
import numpy as np


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculates the Average True Range (ATR) using Wilder's smoothing (RMA).

    Args:
        df: DataFrame containing 'high', 'low', 'close' columns (lowercase).
        period: The lookback period for ATR calculation.

    Returns:
        A pandas Series containing the ATR values.

    Raises:
        ValueError: If required columns are missing or period is invalid.
    """
    required_cols = ["high", "low", "close"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(
            f"DataFrame must contain {required_cols} columns for ATR calculation. Missing: {missing}"
        )
    if not isinstance(period, int) or period <= 0:
        raise ValueError("ATR period must be a positive integer.")

    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())

    tr = pd.DataFrame({"hl": high_low, "hc": high_close, "lc": low_close}).max(axis=1)

    # Calculate ATR using Wilder's smoothing (equivalent to RMA)
    # alpha = 1 / period
    # atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # Alternative calculation using simple moving average for the first value
    # and then exponential smoothing for subsequent values (common implementation)
    atr = (
        tr.rolling(window=period).mean().bfill()
    )  # Use SMA for initial value, backfill NaNs
    atr = atr.ewm(alpha=1 / period, adjust=False).mean()  # Apply Wilder's smoothing

    # For strict Wilder's smoothing (RMA) from scratch:
    # atr = pd.Series(index=df.index, dtype=float)
    # atr.iloc[period - 1] = tr.iloc[:period].mean() # Initial SMA
    # for i in range(period, len(df)):
    #     atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period

    return atr


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA).

    Args:
        series: pandas Series of prices (e.g., 'close').
        period: The lookback period for SMA calculation.

    Returns:
        A pandas Series containing the SMA values.

    Raises:
        ValueError: If period is invalid.
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError("SMA period must be a positive integer.")
    return series.rolling(window=period, min_periods=period).mean()


def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA) using HLC/3 as input.

    Args:
        df: DataFrame containing 'high', 'low', 'close' columns (lowercase).
        period: The lookback period for EMA calculation.

    Returns:
        A pandas Series containing the EMA values.

    Raises:
        ValueError: If required columns are missing or period is invalid.
    """
    required_cols = ["high", "low", "close"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(
            f"DataFrame must contain {required_cols} columns for EMA calculation. Missing: {missing}"
        )
    if not isinstance(period, int) or period <= 0:
        raise ValueError("EMA period must be a positive integer.")

    # Use typical price (HLC/3) for EMA calculation as it's less prone to
    # outlier wicks compared to just Close, especially for trend filtering.
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    return typical_price.ewm(span=period, adjust=False, min_periods=period).mean()
