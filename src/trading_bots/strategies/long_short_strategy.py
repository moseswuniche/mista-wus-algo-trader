import pandas as pd
import numpy as np
from typing import List, Tuple

from .base_strategy import Strategy


class LongShortStrategy(Strategy):
    """
    A strategy that goes long on negative returns and short on positive returns,
    filtered by volume change.
    """

    def __init__(
        self, return_thresh: Tuple[float, float], volume_thresh: Tuple[float, float]
    ) -> None:
        """
        Initializes the LongShortStrategy.

        Args:
            return_thresh: Tuple containing the lower and upper return thresholds.
            volume_thresh: Tuple containing the lower and upper volume change thresholds.
        """
        super().__init__(return_thresh=return_thresh, volume_thresh=volume_thresh)
        self.return_thresh = return_thresh
        self.volume_thresh = volume_thresh

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on return and volume change.

        Args:
            data: DataFrame containing 'Close' and 'Volume' columns.

        Returns:
            A pandas Series containing the position signal (1, -1, or 0).
        """
        df = data[["Close", "Volume"]].copy()
        df["returns"] = np.log(df.Close / df.Close.shift())

        # Calculate volume change, handling potential zero volume
        volume_ratio = df.Volume.div(df.Volume.shift(1))
        # Replace zero or negative ratios with NaN to avoid log errors
        volume_ratio.loc[volume_ratio <= 0] = np.nan
        df["vol_ch"] = np.log(volume_ratio)

        # Cap extreme volume changes (optional, depends on if log already handled extremes)
        # df.loc[df.vol_ch > 3, "vol_ch"] = np.nan
        # df.loc[df.vol_ch < -3, "vol_ch"] = np.nan
        # Drop rows where vol_ch is NaN (due to zero volume or initial shift)
        df.dropna(subset=["vol_ch"], inplace=True)

        cond1 = df.returns <= self.return_thresh[0]
        cond2 = df.vol_ch.between(self.volume_thresh[0], self.volume_thresh[1])
        cond3 = df.returns >= self.return_thresh[1]

        position = pd.Series(index=df.index, dtype=int).fillna(0)
        position.loc[cond1 & cond2] = 1
        position.loc[cond3 & cond2] = -1

        return position
