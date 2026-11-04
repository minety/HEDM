"""K线包含处理 (inclusion merging) and 分型 (fractal) detection, both incremental."""

from __future__ import annotations

from .models import Bar, FxType, Fractal, MergedBar


def _contains(a_high: float, a_low: float, b_high: float, b_low: float) -> bool:
    """True when one range contains the other (包含关系)."""
    return (a_high >= b_high and a_low <= b_low) or (b_high >= a_high and b_low <= a_low)


class KLineProcessor:
    """Turns raw bars into merged bars and confirmed fractals, one bar at a time.

    Merging direction follows the last two merged bars: when the latest merged
    bar's high is above the previous one we are going up and take the higher
    high and the higher low; otherwise we take the lower high and the lower low.
    With only one merged bar the direction defaults to up.

    A fractal on merged bar ``i`` is confirmed as soon as merged bar ``i + 1``
    exists. Later merges into bar ``i + 1`` never undo it: merging keeps moving
    bar ``i + 1`` further away from bar ``i`` in the same direction.
    """

    def __init__(self, bars: list[Bar]):
        self._bars = bars  # shared with the analyzer, read-only here
        self.merged: list[MergedBar] = []
        self.fractals: list[Fractal] = []

    def _going_up(self) -> bool:
        m = self.merged
        if len(m) < 2:
            return True
        return m[-1].high > m[-2].high

    def update(self, bar: Bar) -> Fractal | None:
        """Feed one raw bar; returns a newly confirmed fractal, if any."""
        m = self.merged
        i = bar.index
        if m:
            last = m[-1]
            if _contains(last.high, last.low, bar.high, bar.low):
                if self._going_up():
                    if bar.high > last.high:
                        last.high, last.high_raw = bar.high, i
                    if bar.low > last.low:
                        last.low, last.low_raw = bar.low, i
                else:
                    if bar.low < last.low:
                        last.low, last.low_raw = bar.low, i
                    if bar.high < last.high:
                        last.high, last.high_raw = bar.high, i
                last.end = i
                return None
        m.append(MergedBar(len(m), bar.high, bar.low, i, i, i, i))
        return self._check_fractal(confirm_raw=i)

    def _check_fractal(self, confirm_raw: int) -> Fractal | None:
        m = self.merged
        if len(m) < 3:
            return None
        a, b, c = m[-3], m[-2], m[-1]
        fx: Fractal | None = None
        zone_high = max(a.high, b.high, c.high)
        zone_low = min(a.low, b.low, c.low)
        if b.high > a.high and b.high > c.high:
            fx = Fractal(
                FxType.TOP, b.index, b.high_raw, self._bars[b.high_raw].dt, b.high,
                zone_high, zone_low, confirm_raw,
            )
        elif b.low < a.low and b.low < c.low:
            fx = Fractal(
                FxType.BOTTOM, b.index, b.low_raw, self._bars[b.low_raw].dt, b.low,
                zone_high, zone_low, confirm_raw,
            )
        if fx is not None:
            self.fractals.append(fx)
        return fx
