"""Resample OHLCV bars to a coarser frequency, never mixing trading days."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def filter_session(df: "pd.DataFrame", start: str = "09:30", end: str = "16:00") -> "pd.DataFrame":
    """Keep rows whose bar *start* time is in [start, end). Expects a DatetimeIndex."""
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    tod = df.index.hour * 60 + df.index.minute
    return df[(tod >= sh * 60 + sm) & (tod < eh * 60 + em)]


def resample_ohlcv(df: "pd.DataFrame", rule: str = "5min", offset: str | None = None) -> "pd.DataFrame":
    """Aggregate start-labelled bars (DatetimeIndex; open/high/low/close/volume columns).

    Bins are left-closed and left-labelled, computed per calendar day so a bin
    never spans the overnight gap. ``offset`` shifts the bin grid, e.g.
    ``rule="60min", offset="30min"`` gives 09:30, 10:30, ... hourly bars.
    """
    import pandas as pd

    if df.empty:
        return df
    kw = {"label": "left", "closed": "left"}
    if offset:
        kw["offset"] = offset
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    minutes = pd.Timedelta(rule).total_seconds() / 60
    if minutes > 0 and 1440 % minutes == 0:
        # Bins anchored at midnight tile every day exactly, so one pass is safe.
        out = df.resample(rule, origin="start_day", **kw).agg(agg).dropna(subset=["open"])
    else:
        parts = [day.resample(rule, **kw).agg(agg).dropna(subset=["open"])
                 for _, day in df.groupby(df.index.normalize(), sort=True)]
        out = pd.concat(parts)
    out.index.name = df.index.name or "dt"
    return out
