"""线段 (segment) construction via 特征序列 (characteristic sequence) fractals.

For an up segment the characteristic sequence is its down strokes, each taken
as the price interval it covers. Elements are merged with inclusion handling
in the segment's direction (up: higher high and higher low). A top fractal in
the processed sequence ends the segment at the top's high:

* Case 1 — no gap between the fractal's first and second element: the segment
  ends as soon as the fractal is complete.
* Case 2 — a gap exists: the segment only ends once the characteristic
  sequence of the new (down) segment, i.e. its up strokes, forms a bottom
  fractal. If price trades above the top first, the old segment continues.

Down segments are symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import Bi, Direction, Seg


@dataclass
class _Elem:
    high: float
    low: float
    bi: int  # index of the stroke that supplies the extreme in segment direction


def _contained(a: _Elem, b: _Elem) -> bool:
    return (a.high >= b.high and a.low <= b.low) or (b.high >= a.high and b.low <= a.low)


def _push(elems: list[_Elem], e: _Elem, up: bool) -> None:
    """Append ``e`` with inclusion handling; ``up`` selects the merge direction.

    An element that contains the previous one *and* reaches a new extreme in
    the segment direction is a candidate fractal peak. It is kept separate:
    the first and second elements of a fractal are never merged, otherwise the
    peak would lose its left neighbour.
    """
    if elems and _contained(elems[-1], e):
        last = elems[-1]
        if up and e.high <= last.high:
            elems[-1] = _Elem(last.high, max(last.low, e.low), last.bi)
            return
        if not up and e.low >= last.low:
            elems[-1] = _Elem(min(last.high, e.high), last.low, last.bi)
            return
    elems.append(e)


def _is_peak(elems: list[_Elem], up: bool) -> bool:
    """Fractal at elems[-2] whose peak is also the extreme of the whole sequence.

    The second condition stops a local fractal from ending a segment while a
    more extreme point (e.g. inside the first element) is still in it.
    """
    a, b, c = elems[-3], elems[-2], elems[-1]
    if up:
        return b.high > a.high and b.high > c.high and b.high >= max(e.high for e in elems)
    return b.low < a.low and b.low < c.low and b.low <= min(e.low for e in elems)


@dataclass
class _ScanResult:
    status: str  # "finished" | "unfinished" | "start_broken"
    end_bi: int | None = None
    confirm_bi: int = -1
    break_bi: int = -1  # for "start_broken": stroke that traded beyond the start

    @property
    def finished(self) -> bool:
        return self.status == "finished"


def _second_sequence(bis: Sequence[Bi], k: int, up: bool, peak: float, origin: float) -> tuple[str, int]:
    """Case-2 check. Stroke ``k`` starts the new segment at ``peak``.

    Confirmed when the new segment's characteristic sequence forms a fractal,
    or when the new segment trades beyond ``origin`` (the old segment's start),
    which destroys the old segment outright. Broken when price exceeds the
    peak first. Returns ("confirmed", i), ("broken", i) or ("pending", -1).
    """
    elems: list[_Elem] = []
    for m in range(k, len(bis)):
        b = bis[m]
        if (m - k) % 2 == 0:  # strokes in the new segment's direction
            if (up and b.low < origin) or (not up and b.high > origin):
                return "confirmed", m
            continue
        if (up and b.high > peak) or (not up and b.low < peak):
            return "broken", m
        _push(elems, _Elem(b.high, b.low, m), up=not up)
        if len(elems) >= 3 and _is_peak(elems, up=not up):
            return "confirmed", m
    return "pending", -1


def _scan_segment(bis: Sequence[Bi], s: int, not_before: int = -1, check_start: bool = True) -> _ScanResult:
    """Scan the segment starting at stroke ``s``.

    ``not_before``: ignore terminations whose extreme stroke is at or before
    this index (used after a false termination was undone).
    ``check_start``: report "start_broken" when an opposite stroke trades
    beyond the segment's start before the segment terminates.
    """
    up = bis[s].direction is Direction.UP
    origin = bis[s].start_price
    elems: list[_Elem] = []
    n = len(bis)
    for j in range(s + 1, n, 2):
        b = bis[j]
        if check_start and ((up and b.low < origin) or (not up and b.high > origin)):
            return _ScanResult("start_broken", break_bi=j)
        _push(elems, _Elem(b.high, b.low, j), up=up)
        if len(elems) < 3 or not _is_peak(elems, up=up):
            continue
        first, peak = elems[-3], elems[-2]
        k = peak.bi  # stroke that starts at the segment's extreme
        if k <= not_before:
            continue
        gap = peak.low > first.high if up else peak.high < first.low
        if not gap:
            return _ScanResult("finished", k - 1, j)
        status, at = _second_sequence(bis, k, up, peak.high if up else peak.low, origin)
        if status == "confirmed":
            return _ScanResult("finished", k - 1, max(at, j))
        if status == "pending":
            return _ScanResult("unfinished", k - 1)
        # broken: the old segment goes on, keep scanning
    # Unfinished: the segment currently ends at its most extreme stroke.
    same_dir = range(s, n, 2)
    end = max(same_dir, key=lambda i: bis[i].high) if up else min(same_dir, key=lambda i: bis[i].low)
    return _ScanResult("unfinished", end)


def _valid_start(bis: Sequence[Bi], i: int) -> bool:
    """Initial segment start: the first three strokes must make progress."""
    a, b, c = bis[i], bis[i + 1], bis[i + 2]
    if a.direction is Direction.UP:
        return b.low > a.low and c.high > a.high
    return b.high < a.high and c.low < a.low


def _make_seg(bis: Sequence[Bi], index: int, s: int, e: int, finished: bool, confirm: int) -> Seg:
    a, b = bis[s], bis[e]
    return Seg(
        index=index,
        direction=a.direction,
        start_price=a.start_price,
        end_price=b.end_price,
        start_raw=a.start_raw,
        end_raw=b.end_raw,
        start_dt=a.start_dt,
        end_dt=b.end_dt,
        start_bi=s,
        end_bi=e,
        finished=finished,
        confirm_bi=confirm,
    )


def build_segments(bis: Sequence[Bi], previous: Sequence[Seg] = (), first_changed_bi: int = 0) -> list[Seg]:
    """Build segments from strokes.

    ``previous``/``first_changed_bi`` allow incremental use: finished segments
    whose confirmation only used strokes before ``first_changed_bi`` are kept.
    """
    segs: list[Seg] = []
    for sg in previous:
        if sg.finished and sg.confirm_bi < first_changed_bi:
            segs.append(sg)
        else:
            break

    n = len(bis)

    def first_valid_start(i: int) -> int | None:
        return next((k for k in range(i, n - 2) if _valid_start(bis, k)), None)

    start = segs[-1].end_bi + 1 if segs else first_valid_start(0)
    not_before: dict[int, int] = {}

    while start is not None and start + 2 < n:
        initial = not segs
        res = _scan_segment(bis, start, not_before.get(start, -1), check_start=True)
        if res.status == "start_broken":
            if initial:
                # The chosen first segment was wrong; try a later start.
                start = first_valid_start(start + 1)
                continue
            # The previous segment ended too early: extend it past the break.
            prev = segs.pop()
            not_before[prev.start_bi] = max(not_before.get(prev.start_bi, -1), res.break_bi)
            start = prev.start_bi
            continue
        if res.end_bi is None or res.end_bi - start < 2:
            break
        segs.append(_make_seg(bis, len(segs), start, res.end_bi, res.finished, res.confirm_bi))
        if not res.finished:
            break
        start = res.end_bi + 1
    return segs
