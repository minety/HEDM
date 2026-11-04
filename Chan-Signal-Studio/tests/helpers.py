"""Test helpers: build bars that follow a scripted zig-zag path."""

from __future__ import annotations

from datetime import datetime, timedelta

from chan_signal_studio.models import Bi, Direction, Bar

WICK = 0.05


def zigzag_bars(pivots, legs=6, start=datetime(2024, 1, 1), wick=WICK, step=timedelta(days=1)):
    """Linear legs between ``pivots``; ``legs`` is bars per leg (int or list)."""
    prices = [float(pivots[0])]
    for i in range(len(pivots) - 1):
        n = legs[i] if isinstance(legs, (list, tuple)) else legs
        a, b = pivots[i], pivots[i + 1]
        prices += [a + (b - a) * k / n for k in range(1, n + 1)]
    bars = []
    for i in range(1, len(prices)):
        o, c = prices[i - 1], prices[i]
        bars.append(Bar(start + step * i, o, max(o, c) + wick, min(o, c) - wick, c, 1000.0))
    return bars


def make_bis(pivots):
    """Stroke objects straight from pivot prices (one raw bar per pivot)."""
    t0 = datetime(2024, 1, 1)
    out = []
    for i in range(len(pivots) - 1):
        a, b = pivots[i], pivots[i + 1]
        out.append(Bi(
            index=i, direction=Direction.UP if b > a else Direction.DOWN,
            start_price=a, end_price=b, start_raw=i * 5, end_raw=(i + 1) * 5,
            start_dt=t0 + timedelta(days=i * 5), end_dt=t0 + timedelta(days=(i + 1) * 5),
        ))
    return out
