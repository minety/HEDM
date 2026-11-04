"""三类买卖点 detection on one level (strokes or segments).

Detection only looks at the newest move, right after the structure changed,
so every signal carries the bar on which it became known and is never
revised afterwards. A later structural change (the move extending further)
simply produces a new signal at the new point; strategies protect themselves
with the signal's ``stop_price``.

* B1 / S1 (第一类): the newest move leaves the latest pivot to a new extreme,
  in the same direction as the move that entered the pivot, and is weaker
  than that entering move (背驰). ``trend`` is True when the previous pivot
  lies entirely beyond the current one in the opposite direction (趋势背驰),
  otherwise it is a 盘整背驰.
* B2 / S2 (第二类): the first pullback after a B1/S1 that does not reach the
  B1/S1 price.
* B3 / S3 (第三类): after a pivot (at least three moves plus a leaving move),
  the return move fails to get back into [ZD, ZG].
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from .config import ChanConfig
from .indicators import IncrementalMACD
from .models import Direction, Move, Signal, SignalKind, ZhongShu


class SignalDetector:
    def __init__(self, config: ChanConfig, macd: IncrementalMACD):
        self.config = config
        self.macd = macd
        # One B2/S2 per first-class point and one B3/S3 per pivot, even when
        # the pullback that produced it keeps extending.
        self._once: set[tuple] = set()

    # ------------------------------------------------------------- 背驰
    def divergence(self, entry: Move, leave: Move) -> tuple[bool, dict]:
        up = leave.direction is Direction.UP
        ratio = self.config.divergence_ratio
        checks: dict[str, dict] = {}
        for method in self.config.divergence_methods:
            if method == "area":
                e = self.macd.area(entry.start_raw, entry.end_raw, up)
                lv = self.macd.area(leave.start_raw, leave.end_raw, up)
                ok = lv < ratio * e
            elif method == "dif":
                e = self.macd.dif_extreme(entry.start_raw, entry.end_raw, up)
                lv = self.macd.dif_extreme(leave.start_raw, leave.end_raw, up)
                ok = lv < e
            else:  # slope
                e = entry.amplitude / max(entry.length, 1)
                lv = leave.amplitude / max(leave.length, 1)
                ok = lv < ratio * e
            checks[method] = {"entry": e, "leave": lv, "diverged": ok}
        flags = [c["diverged"] for c in checks.values()]
        passed = any(flags) if self.config.divergence_mode == "any" else all(flags)
        return passed, checks

    # ------------------------------------------------------------- detect
    def detect(
        self,
        level: str,
        moves: Sequence[Move],
        pivots: Sequence[ZhongShu],
        known: dict[tuple[str, str, int], Signal],
        detect_raw: int,
        detect_dt: datetime,
        nested: Signal | None = None,
    ) -> list[Signal]:
        """Signals on the newest move. ``nested`` is the lower level's type-1
        candidate at its newest move (used for 区间套 confirmation)."""
        if len(moves) < 3:
            return []
        self._nested = nested
        out: list[Signal] = []
        for fn in (self._type1, self._type2, self._type3):
            sig = fn(level, moves, pivots, known, detect_raw, detect_dt)
            if sig is None or sig.key in known:
                continue
            once = sig.info.get("once")
            if once is not None:
                if once in self._once:
                    continue
                self._once.add(once)
            out.append(sig)
        return out

    def _signal(self, kind: SignalKind, level: str, m: Move, detect_raw: int, detect_dt: datetime,
                stop: float, **kw) -> Signal:
        return Signal(
            kind=kind, level=level, move_index=m.index, raw_index=m.end_raw, dt=m.end_dt,
            price=m.end_price, detect_raw=detect_raw, detect_dt=detect_dt, stop_price=stop, **kw,
        )

    def _type1(self, level, moves, pivots, known, detect_raw, detect_dt) -> Signal | None:
        if not pivots:
            return None
        z = pivots[-1]
        s = moves[-1]
        if z.finished or z.end != s.index or s.index - z.start < 3:
            return None
        limit = self.config.max_pivot_moves_type1
        if limit and s.index - z.start > limit:
            return None  # the pivot has upgraded to the next level
        entry = moves[z.start - 1]
        if entry.direction is not s.direction:
            return None
        down = s.direction is Direction.DOWN
        comps = moves[z.start : s.index]
        if down:
            if not (entry.high > z.zg and s.low < z.zd and s.low < entry.low
                    and s.low < min(m.low for m in comps)):
                return None
        else:
            if not (entry.low < z.zd and s.high > z.zg and s.high > entry.high
                    and s.high > max(m.high for m in comps)):
                return None
        diverged, checks = self.divergence(entry, s)
        if not diverged:
            return None
        trend = False
        if len(pivots) >= 2:
            p = pivots[-2]
            if p.entry_direction is s.direction:
                trend = p.zd > z.zg if down else p.zg < z.zd
        kind = SignalKind.B1 if down else SignalKind.S1
        info = {"entry_move": entry.index, "divergence": checks}
        if level == "seg" and self.config.seg_type1_nested:
            n = getattr(self, "_nested", None)
            if n is None or n.kind is not kind or n.raw_index != s.end_raw:
                return None
            info["nested_bi"] = n.info.get("divergence")
        return self._signal(kind, level, s, detect_raw, detect_dt, stop=s.end_price,
                            zs_index=z.index, trend=trend, info=info)

    def type1_candidate(self, level, moves, pivots) -> Signal | None:
        """Stateless type-1 check on the newest move (no dedupe, nothing recorded)."""
        if len(moves) < 3:
            return None
        return self._type1(level, moves, pivots, {}, -1, None)

    def _type2(self, level, moves, pivots, known, detect_raw, detect_dt) -> Signal | None:
        s = moves[-1]
        prior = moves[-3]
        down = s.direction is Direction.DOWN
        want = SignalKind.B1 if down else SignalKind.S1
        first = known.get((level, want.value, prior.end_raw))
        if first is None or first.move_index != prior.index:
            return None
        if down and not s.low > prior.low:
            return None
        if not down and not s.high < prior.high:
            return None
        kind = SignalKind.B2 if down else SignalKind.S2
        return self._signal(kind, level, s, detect_raw, detect_dt, stop=first.price,
                            zs_index=first.zs_index,
                            info={"first_point": first.raw_index,
                                  "once": (level, kind.value, first.raw_index)})

    def _type3(self, level, moves, pivots, known, detect_raw, detect_dt) -> Signal | None:
        if not pivots:
            return None
        z = pivots[-1]
        s = moves[-1]
        if not z.finished or z.end != s.index - 1 or z.end - z.start < 3:
            return None
        once = (level, z.start_raw, z.zg, z.zd)
        if s.direction is Direction.DOWN and s.low > z.zg:
            return self._signal(SignalKind.B3, level, s, detect_raw, detect_dt, stop=z.zg,
                                zs_index=z.index, info={"zg": z.zg, "zd": z.zd, "once": once})
        if s.direction is Direction.UP and s.high < z.zd:
            return self._signal(SignalKind.S3, level, s, detect_raw, detect_dt, stop=z.zd,
                                zs_index=z.index, info={"zg": z.zg, "zd": z.zd, "once": once})
        return None
