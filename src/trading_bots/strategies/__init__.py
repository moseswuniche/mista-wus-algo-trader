# flake8: noqa
# Import base class
from .base_strategy import BaseStrategy

# Import all strategy classes defined by the new config
from .ma_crossover_strategy import MovingAverageCrossoverStrategy
from .scalping import ScalpingStrategy
from .bb_reversion_strategy import BollingerBandReversionStrategy
from .momentum import MomentumStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .breakout_strategy import BreakoutStrategy
from .hybrid_strategy import HybridStrategy

# Define __all__ for explicit export
__all__ = [
    "BaseStrategy",
    "MovingAverageCrossoverStrategy",
    "ScalpingStrategy",
    "BollingerBandReversionStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "BreakoutStrategy",
    "HybridStrategy",
]
