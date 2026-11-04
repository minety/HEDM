"""Reference strategy: trade the 三类买卖点 with structural stops."""

from __future__ import annotations

from dataclasses import dataclass

from .base import Strategy, StrategyContext


@dataclass
class ChanStrategyParams:
    levels: tuple[str, ...] = ("bi",)  # which levels' signals to trade
    buy_kinds: tuple[str, ...] = ("B1", "B2", "B3")  # entries (long)
    sell_kinds: tuple[str, ...] = ("S1", "S2", "S3")  # exits for longs / entries for shorts
    position_pct: float = 1.0  # fraction of equity per position
    use_stop: bool = True  # exit when the close crosses the signal's stop price
    allow_short: bool = False  # also open shorts on sell signals (market must allow it)
    require_trend_b1: bool = False  # only take B1/S1 from trend divergence


class ChanStrategy(Strategy):
    """Long on buy points, flat (or short) on sell points.

    Stops are structural: B1 -> the B1 low, B2 -> the B1 low, B3 -> the pivot's
    ZG; mirrored for sells. Close-based, executed at the next open.
    """

    def __init__(self, params: ChanStrategyParams | None = None, **kw):
        self.p = params or ChanStrategyParams(**kw)
        self.side = 0  # intended side after pending orders: 1 long, -1 short, 0 flat
        self.stop: float | None = None

    def _accept(self, sig) -> bool:
        if sig.level not in self.p.levels:
            return False
        if self.p.require_trend_b1 and sig.kind.order == 1 and not sig.trend:
            return False
        return True

    def on_bar(self, ctx: StrategyContext) -> None:
        p = self.p
        if ctx.pending is None:
            # Re-sync with the broker in case an order was dropped (e.g. not enough cash).
            actual = (ctx.position > 0) - (ctx.position < 0)
            if actual != self.side:
                self.side = actual
                if actual == 0:
                    self.stop = None
        for sig in ctx.new_signals:
            if not self._accept(sig):
                continue
            k = sig.kind.value
            if sig.is_buy:
                if self.side < 0:
                    ctx.close_position(f"cover on {sig.level}:{k}")
                    self.side, self.stop = 0, None
                if k in p.buy_kinds and self.side == 0:
                    ctx.order_target_percent(p.position_pct, f"{sig.level}:{k}")
                    self.side, self.stop = 1, sig.stop_price
            else:
                if self.side > 0 and k in p.sell_kinds:
                    ctx.close_position(f"exit on {sig.level}:{k}")
                    self.side, self.stop = 0, None
                if p.allow_short and k in p.sell_kinds and self.side == 0:
                    ctx.order_target_percent(-p.position_pct, f"{sig.level}:{k}")
                    self.side, self.stop = -1, sig.stop_price

        if p.use_stop and self.stop is not None:
            c = ctx.bar.close
            if self.side > 0 and c < self.stop:
                ctx.close_position("stop")
                self.side, self.stop = 0, None
            elif self.side < 0 and c > self.stop:
                ctx.close_position("stop")
                self.side, self.stop = 0, None
