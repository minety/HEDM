"""Chan-Signal-Studio: a 缠论 (Chan theory) analysis and trading framework."""

__version__ = "0.1.0"

from .analyzer import ChanAnalyzer, analyze
from .config import ChanConfig
from .models import Bar, Bi, Direction, Fractal, FxType, MergedBar, Seg, Signal, SignalKind, ZhongShu

__all__ = [
    "ChanAnalyzer", "analyze", "ChanConfig",
    "Bar", "Bi", "Direction", "Fractal", "FxType", "MergedBar", "Seg", "Signal", "SignalKind", "ZhongShu",
    "__version__",
]
