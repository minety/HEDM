"""Strategy interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..models import Bar, Signal

if TYPE_CHECKING:  # pragma: no cover
    from ..analyzer import ChanAnalyzer
    from ..backtest.broker import Broker


@dataclass
class StrategyContext:
    """What a strategy sees on each bar (after the bar has closed).

    Orders are executed at the next bar's open, so nothing a strategy does can
    use information from the future.
    """

    bar: Bar
    index: int
    analyzer: "ChanAnalyzer"
    broker: "Broker"
    new_signals: list[Signal]
    pending: tuple[float, str] | None = None  # order still waiting to fill, if any
    _order: tuple[float, str] | None = field(default=None, repr=False)

    @property
    def position(self) -> float:
        return self.broker.qty

    @property
    def equity(self) -> float:
        return self.broker.equity(self.bar.close)

    def order_target_percent(self, pct: float, reason: str = "") -> None:
        """Target position as a fraction of equity (negative = short). Fills next bar."""
        self._order = (float(pct), reason)

    def close_position(self, reason: str = "close") -> None:
        self.order_target_percent(0.0, reason)


class Strategy:
    """Subclass and implement :meth:`on_bar`."""

    def on_start(self, analyzer: "ChanAnalyzer") -> None:  # pragma: no cover - optional hook
        pass

    def on_bar(self, ctx: StrategyContext) -> None:
        raise NotImplementedError
