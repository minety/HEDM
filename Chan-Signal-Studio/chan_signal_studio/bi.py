"""笔 (stroke) construction from confirmed fractals, incremental."""

from __future__ import annotations

from .models import Bar, Bi, Direction, FxType, Fractal, MergedBar


class BiBuilder:
    """Maintains the list of stroke endpoints as fractals arrive.

    Rules, applied to each new fractal ``f`` against the last endpoint ``p``:

    1. Same type and more extreme -> ``f`` replaces ``p`` (the stroke extends).
    2. Opposite type and a valid stroke ``p -> f`` -> ``f`` becomes a new endpoint.
       Valid means: enough bars between them (``bi_mode``), the two fractal
       zones do not overlap in price, and ``f`` is the extreme of every merged
       bar between ``p`` and ``f``.
    3. Opposite type, not a valid stroke, but ``f`` is beyond the start of the
       last stroke -> the last stroke was premature. Drop ``p`` and let ``f``
       extend the stroke before it, as long as ``p`` did not exceed that
       stroke's start (otherwise the structure is kept as is).

    Only the last two endpoints can ever change, so every stroke except the
    last one or two is final.
    """

    def __init__(self, bars: list[Bar], merged: list[MergedBar], mode: str = "new"):
        self._bars = bars
        self._merged = merged
        self.mode = mode
        self.points: list[Fractal] = []
        self.bis: list[Bi] = []

    # ------------------------------------------------------------------ rules
    def _enough_distance(self, a: Fractal, b: Fractal) -> bool:
        if self.mode == "old":
            return b.index - a.index >= 4
        return b.index - a.index >= 3 and b.raw_index - a.raw_index >= 4

    def _is_extreme_end(self, a: Fractal, b: Fractal) -> bool:
        span = self._merged[a.index : b.index + 1]
        if b.type is FxType.TOP:
            return b.price >= max(m.high for m in span)
        return b.price <= min(m.low for m in span)

    def _valid_bi(self, a: Fractal, b: Fractal) -> bool:
        if a.type is b.type or not self._enough_distance(a, b):
            return False
        if b.type is FxType.TOP:
            price_ok = b.price > a.zone_high and a.price < b.zone_low
        else:
            price_ok = b.price < a.zone_low and a.price > b.zone_high
        return price_ok and self._is_extreme_end(a, b)

    # ----------------------------------------------------------------- update
    def add(self, f: Fractal) -> int | None:
        """Feed a fractal. Returns the index of the first stroke that changed, or None."""
        pts = self.points
        if not pts:
            pts.append(f)
            return None
        last = pts[-1]
        changed_point: int | None = None
        if f.type is last.type:
            if f.more_extreme_than(last):
                pts[-1] = f
                changed_point = len(pts) - 1
        elif self._valid_bi(last, f):
            pts.append(f)
            changed_point = len(pts) - 1
        elif len(pts) >= 2 and f.more_extreme_than(pts[-2]):
            # f broke the start of the last stroke before a new stroke formed.
            if len(pts) == 2 or not last.more_extreme_than(pts[-3]):
                pts.pop()
                pts[-1] = f
                changed_point = len(pts) - 1
        if changed_point is None:
            return None
        first_bi = max(changed_point - 1, 0)
        self._rebuild_from(first_bi)
        return first_bi

    def _rebuild_from(self, k: int) -> None:
        del self.bis[k:]
        pts = self.points
        for i in range(k, len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            self.bis.append(
                Bi(
                    index=i,
                    direction=Direction.UP if b.type is FxType.TOP else Direction.DOWN,
                    start_price=a.price,
                    end_price=b.price,
                    start_raw=a.raw_index,
                    end_raw=b.raw_index,
                    start_dt=a.dt,
                    end_dt=b.dt,
                    start_fx=a,
                    end_fx=b,
                )
            )
