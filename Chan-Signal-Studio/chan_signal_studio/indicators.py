"""Incremental MACD with prefix sums for fast per-move area queries."""

from __future__ import annotations


class IncrementalMACD:
    """MACD using the common Chinese convention: hist = 2 * (DIF - DEA)."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        if not (fast > 0 and slow > fast and signal > 0):
            raise ValueError("require 0 < fast < slow and signal > 0")
        self._af = 2.0 / (fast + 1)
        self._as = 2.0 / (slow + 1)
        self._ad = 2.0 / (signal + 1)
        self._ema_f: float | None = None
        self._ema_s: float | None = None
        self._dea: float | None = None
        self.dif: list[float] = []
        self.dea: list[float] = []
        self.hist: list[float] = []
        # prefix sums of positive and negative histogram parts
        self._pos = [0.0]
        self._neg = [0.0]

    def update(self, close: float) -> None:
        if self._ema_f is None:
            self._ema_f = self._ema_s = close
            self._dea = 0.0
        else:
            self._ema_f += self._af * (close - self._ema_f)
            self._ema_s += self._as * (close - self._ema_s)
        dif = self._ema_f - self._ema_s
        self._dea += self._ad * (dif - self._dea)
        h = 2.0 * (dif - self._dea)
        self.dif.append(dif)
        self.dea.append(self._dea)
        self.hist.append(h)
        self._pos.append(self._pos[-1] + max(h, 0.0))
        self._neg.append(self._neg[-1] + max(-h, 0.0))

    def area(self, start: int, end: int, up: bool) -> float:
        """Histogram area over raw bars [start, end] on the side matching the move."""
        if start > end:
            start, end = end, start
        start = max(start, 0)
        end = min(end, len(self.hist) - 1)
        if end < start:
            return 0.0
        pref = self._pos if up else self._neg
        return pref[end + 1] - pref[start]

    def dif_extreme(self, start: int, end: int, up: bool) -> float:
        """DIF extreme in the move's direction, signed so that larger = stronger.

        Up moves return max(DIF); down moves return -min(DIF).
        """
        seg = self.dif[max(start, 0) : end + 1]
        if not seg:
            return 0.0
        return max(seg) if up else -min(seg)
