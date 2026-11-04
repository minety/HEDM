"""Core data structures used across Chan-Signal-Studio.

术语对照 (Chinese term ↔ class):

* K线 (raw bar)               -> :class:`Bar`
* 包含处理后的K线 (merged bar)  -> :class:`MergedBar`
* 分型 (fractal)               -> :class:`Fractal`
* 笔 (stroke)                  -> :class:`Bi`
* 线段 (segment)               -> :class:`Seg`
* 中枢 (pivot / central zone)   -> :class:`ZhongShu`
* 买卖点 (buy/sell point)       -> :class:`Signal`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"

    @property
    def opposite(self) -> "Direction":
        return Direction.DOWN if self is Direction.UP else Direction.UP

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class FxType(str, Enum):
    TOP = "top"  # 顶分型
    BOTTOM = "bottom"  # 底分型

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class SignalKind(str, Enum):
    B1 = "B1"  # 第一类买点
    B2 = "B2"  # 第二类买点
    B3 = "B3"  # 第三类买点
    S1 = "S1"  # 第一类卖点
    S2 = "S2"  # 第二类卖点
    S3 = "S3"  # 第三类卖点

    @property
    def is_buy(self) -> bool:
        return self.value.startswith("B")

    @property
    def order(self) -> int:
        return int(self.value[1])

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


@dataclass(frozen=True)
class Bar:
    """One raw OHLCV candle. ``index`` is assigned by the analyzer."""

    dt: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    index: int = -1

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"bar at {self.dt}: high {self.high} < low {self.low}")


@dataclass
class MergedBar:
    """A bar after 包含处理 (inclusion merging); may span several raw bars."""

    index: int  # position in the merged list
    high: float
    low: float
    start: int  # first raw bar index
    end: int  # last raw bar index
    high_raw: int  # raw index of the bar that supplies ``high``
    low_raw: int  # raw index of the bar that supplies ``low``


@dataclass(frozen=True)
class Fractal:
    """顶分型 / 底分型 on merged bars."""

    type: FxType
    index: int  # merged index of the middle bar
    raw_index: int  # raw index of the extreme bar
    dt: datetime
    price: float  # high for a top, low for a bottom
    zone_high: float  # highest high of the three merged bars
    zone_low: float  # lowest low of the three merged bars
    confirm_raw: int  # raw index of the bar on which the fractal became known

    def more_extreme_than(self, other: "Fractal") -> bool:
        if self.type is FxType.TOP:
            return self.price > other.price
        return self.price < other.price


@dataclass
class Move:
    """A directed price move between two points. Base class of :class:`Bi` and :class:`Seg`."""

    index: int
    direction: Direction
    start_price: float
    end_price: float
    start_raw: int
    end_raw: int
    start_dt: datetime
    end_dt: datetime

    @property
    def high(self) -> float:
        return max(self.start_price, self.end_price)

    @property
    def low(self) -> float:
        return min(self.start_price, self.end_price)

    @property
    def length(self) -> int:
        """Number of raw bars spanned (end - start)."""
        return self.end_raw - self.start_raw

    @property
    def amplitude(self) -> float:
        return abs(self.end_price - self.start_price)


@dataclass
class Bi(Move):
    """笔: connects a bottom and a top fractal."""

    start_fx: Fractal | None = None
    end_fx: Fractal | None = None


@dataclass
class Seg(Move):
    """线段: built from at least three strokes, terminated by 特征序列分型."""

    start_bi: int = 0
    end_bi: int = 0
    finished: bool = False
    # Highest stroke index that had to be inspected to confirm the end. Used to
    # decide which segments stay valid when the newest strokes are revised.
    confirm_bi: int = -1


@dataclass
class ZhongShu:
    """中枢: overlap zone of at least three consecutive moves of one level."""

    level: str  # "bi" or "seg"
    index: int
    start: int  # index of the first component move
    end: int  # index of the last move that still touches [zd, zg]
    zg: float  # 中枢上沿 = min of the first three highs
    zd: float  # 中枢下沿 = max of the first three lows
    gg: float  # highest high of the components
    dd: float  # lowest low of the components
    start_raw: int
    end_raw: int
    start_dt: datetime
    end_dt: datetime
    entry_direction: Direction  # direction of the move that entered the zone
    finished: bool = False  # True once a move left without coming back

    @property
    def move_count(self) -> int:
        return self.end - self.start + 1

    def overlaps(self, other: "ZhongShu") -> bool:
        return self.zd <= other.zg and other.zd <= self.zg


@dataclass
class Signal:
    """A buy/sell point detected in real time (never repainted).

    ``raw_index``/``dt``/``price`` describe the structural point (the stroke or
    segment end). ``detect_raw``/``detect_dt`` describe the bar on which the
    point became known, which is when a strategy may act on it.
    """

    kind: SignalKind
    level: str
    move_index: int
    raw_index: int
    dt: datetime
    price: float
    detect_raw: int
    detect_dt: datetime
    stop_price: float
    zs_index: int | None = None
    trend: bool = False  # for type-1 points: trend divergence (True) vs. 盘整背驰
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.kind.is_buy

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.level, self.kind.value, self.raw_index)

    def describe(self) -> str:
        side = "买点" if self.is_buy else "卖点"
        extra = ""
        if self.kind.order == 1:
            extra = " 趋势背驰" if self.trend else " 盘整背驰"
        return (
            f"[{self.level}] {self.kind.value} 第{self.kind.order}类{side}{extra} @ {self.price:.4f} "
            f"({self.dt}) 确认于 {self.detect_dt}, 止损 {self.stop_price:.4f}"
        )
