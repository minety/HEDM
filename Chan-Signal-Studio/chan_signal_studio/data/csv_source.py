"""Load bars from CSV files."""

from __future__ import annotations

from pathlib import Path

from ..models import Bar
from .base import DataSource, bars_from_df


def load_csv(path: str | Path, start: str | None = None, end: str | None = None, **read_kw) -> list[Bar]:
    """Read a CSV with OHLC(V) columns. English or Chinese headers both work."""
    import pandas as pd

    df = pd.read_csv(path, **read_kw)
    bars = bars_from_df(df)
    if start is not None:
        s = pd.Timestamp(start).to_pydatetime()
        bars = [b for b in bars if b.dt >= s]
    if end is not None:
        e = pd.Timestamp(end).to_pydatetime()
        bars = [b for b in bars if b.dt <= e]
    return bars


class CSVSource(DataSource):
    """Looks up ``<root>/<symbol>_<freq>.csv`` or ``<root>/<symbol>.csv``."""

    def __init__(self, root: str | Path = "."):
        self.root = Path(root)

    def get_bars(self, symbol: str, freq: str = "1d", start: str | None = None,
                 end: str | None = None) -> list[Bar]:
        for name in (f"{symbol}_{freq}.csv", f"{symbol}.csv"):
            p = self.root / name
            if p.exists():
                return load_csv(p, start, end)
        raise FileNotFoundError(f"no CSV for {symbol!r} ({freq}) under {self.root}")
