from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    """
    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the strategy with given parameters.
        """
        self.params: Dict[str, Any] = kwargs

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on the input data.

        Args:
            data: DataFrame containing historical or live market data.

        Returns:
            A pandas Series containing the position signal (1 for long, -1 for short, 0 for neutral).
        """
        pass 