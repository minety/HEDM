"""Synthetic OHLC data for demos and tests (no network needed)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from ..models import Bar
from .base import DataSource


def generate_bars(
    n: int = 1500,
    seed: int = 7,
    start: str = "2018-01-02",
    start_price: float = 100.0,
    vol: float = 0.018,
    step: timedelta = timedelta(days=1),
    skip_weekends: bool = True,
) -> list[Bar]:
    """Regime-switching random walk: trending and ranging stretches alternate.

    Regimes last 40-160 bars; drift is drawn per regime so the series contains
    up trends, down trends and consolidations, which is what Chan structures
    need to show up.
    """
    rng = np.random.default_rng(seed)
    rets = np.empty(n)
    i = 0
    while i < n:
        length = int(rng.integers(40, 160))
        regime = rng.choice(["up", "down", "range"], p=[0.38, 0.32, 0.30])
        drift = {"up": 0.0025, "down": -0.0025, "range": 0.0}[regime] * rng.uniform(0.5, 1.5)
        v = vol * rng.uniform(0.7, 1.3)
        rets[i : i + length] = drift + v * rng.standard_normal(min(length, n - i))
        i += length

    closes = start_price * np.exp(np.cumsum(rets))
    t = datetime.fromisoformat(start)
    bars: list[Bar] = []
    prev_close = start_price
    for k in range(n):
        while skip_weekends and step >= timedelta(days=1) and t.weekday() >= 5:
            t += step
        c = float(closes[k])
        o = float(prev_close * np.exp(rng.normal(0, vol * 0.25)))
        wick = vol * 0.6
        h = max(o, c) * float(np.exp(abs(rng.normal(0, wick))))
        lo = min(o, c) * float(np.exp(-abs(rng.normal(0, wick))))
        bars.append(Bar(t, round(o, 4), round(h, 4), round(lo, 4), round(c, 4),
                        float(rng.integers(1_000, 100_000))))
        prev_close = c
        t += step
    return bars


class SyntheticSource(DataSource):
    def __init__(self, n: int = 1500, seed: int = 7):
        self.n = n
        self.seed = seed

    def get_bars(self, symbol: str = "DEMO", freq: str = "1d", start: str | None = None,
                 end: str | None = None) -> list[Bar]:
        # Use the symbol to vary the series deterministically.
        seed = self.seed + sum(map(ord, symbol))
        bars = generate_bars(self.n, seed=seed, start=start or "2018-01-02")
        if end is not None:
            e = datetime.fromisoformat(end)
            bars = [b for b in bars if b.dt <= e]
        return bars
