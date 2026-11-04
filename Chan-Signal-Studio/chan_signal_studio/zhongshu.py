"""中枢 (pivot zone) construction, generic over strokes or segments."""

from __future__ import annotations

from typing import Sequence

from .models import Direction, Move, ZhongShu


def _touches(m: Move, zg: float, zd: float) -> bool:
    return m.low <= zg and m.high >= zd


def _make_zs(moves: Sequence[Move], level: str, index: int, i: int, last: int,
             zg: float, zd: float, finished: bool) -> ZhongShu:
    comps = list(moves[i : last + 1])
    # The final touching move is the leaving move when the zone is finished;
    # keep it out of GG/DD unless the zone would drop below three moves.
    core = comps[:-1] if finished and len(comps) > 3 else comps
    return ZhongShu(
        level=level,
        index=index,
        start=i,
        end=last,
        zg=zg,
        zd=zd,
        gg=max(m.high for m in core),
        dd=min(m.low for m in core),
        start_raw=comps[0].start_raw,
        end_raw=comps[-1].end_raw,
        start_dt=comps[0].start_dt,
        end_dt=comps[-1].end_dt,
        entry_direction=moves[i - 1].direction,
        finished=finished,
    )


def build_zhongshu(
    moves: Sequence[Move],
    level: str,
    require_entry: bool = True,
    previous: Sequence[ZhongShu] = (),
    first_changed: int = 0,
) -> list[ZhongShu]:
    """Find pivots in a sequence of alternating moves.

    A pivot starts at move ``i`` (``i >= 1`` so there is an entering move
    ``i - 1``) when moves ``i, i+1, i+2`` overlap: ZG = min of their highs,
    ZD = max of their lows, ZG > ZD. It extends while following moves still
    touch [ZD, ZG]. The first move that does not touch it finishes the pivot;
    that move is the "return" move used for third-class buy/sell points, and
    the search for the next pivot starts there.

    Finished pivots whose return move lies before ``first_changed`` are reused
    from ``previous``.
    """
    out: list[ZhongShu] = []
    for z in previous:
        if z.finished and z.end + 1 < first_changed:
            out.append(z)
        else:
            break
    i = out[-1].end + 1 if out else 1
    n = len(moves)
    while i + 2 < n:
        a, b, c = moves[i], moves[i + 1], moves[i + 2]
        zg = min(a.high, b.high, c.high)
        zd = max(a.low, b.low, c.low)
        if zg <= zd:
            i += 1
            continue
        if require_entry:
            entry = moves[i - 1]
            if entry.direction is Direction.DOWN and not entry.high > zg:
                i += 1
                continue
            if entry.direction is Direction.UP and not entry.low < zd:
                i += 1
                continue
        j = i + 3
        while j < n and _touches(moves[j], zg, zd):
            j += 1
        finished = j < n
        out.append(_make_zs(moves, level, len(out), i, j - 1, zg, zd, finished))
        if not finished:
            break
        i = j
    return out
