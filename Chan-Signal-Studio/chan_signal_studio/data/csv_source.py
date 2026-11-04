"""Load bars from CSV / TXT files (optionally .gz, .zip, .bz2, .xz compressed)."""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import Bar
from .base import KNOWN_NAMES, DataSource, bars_from_df, normalize_columns

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

# Column layouts for files without a header row, keyed by column count.
HEADERLESS_LAYOUTS = {
    5: ["datetime", "open", "high", "low", "close"],
    6: ["datetime", "open", "high", "low", "close", "volume"],
    7: ["date", "time", "open", "high", "low", "close", "volume"],
    8: ["date", "time", "open", "high", "low", "close", "volume", "open_interest"],
}


def _first_lines(path: Path, n: int = 3) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [i for i in z.namelist() if not i.endswith("/")]
            raw = z.open(names[0])
    elif suffix == ".gz":
        raw = gzip.open(path)
    elif suffix == ".bz2":
        raw = bz2.open(path)
    elif suffix in (".xz", ".lzma"):
        raw = lzma.open(path)
    else:
        raw = open(path, "rb")
    with raw, io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace") as fh:
        return [fh.readline().strip() for _ in range(n)]


def _sniff(path: Path) -> tuple[str, bool]:
    """Return (delimiter, has_header) from the first lines."""
    lines = [ln for ln in _first_lines(path) if ln]
    if not lines:
        raise ValueError(f"{path} is empty")
    first = lines[0]
    sep = max([",", "\t", ";", "|"], key=first.count)
    if first.count(sep) == 0:
        sep = r"\s+"
        tokens = first.split()
    else:
        tokens = first.split(sep)
    names = {t.strip().strip('"').strip("'").lower() for t in tokens}
    return sep, bool(names & KNOWN_NAMES)


def read_ohlcv_csv(path: str | Path, tz: str | None = None, source_tz: str | None = None,
                   **read_kw) -> "pd.DataFrame":
    """Read a price file into a normalized DataFrame (dt, open, high, low, close, volume)."""
    import pandas as pd

    path = Path(path)
    sep, has_header = _sniff(path)
    kw = {"sep": sep, "engine": "python" if sep == r"\s+" else "c", **read_kw}
    if has_header:
        df = pd.read_csv(path, **kw)
    else:
        df = pd.read_csv(path, header=None, **kw)
        layout = HEADERLESS_LAYOUTS.get(df.shape[1])
        if layout is None:
            raise ValueError(f"{path}: no header and {df.shape[1]} columns; add a header row "
                             "(e.g. datetime,open,high,low,close,volume)")
        if df.shape[1] == 7 and not _looks_like_time(df.iloc[:, 1]):
            layout = ["datetime", "open", "high", "low", "close", "volume", "open_interest"]
        df.columns = layout
    return normalize_columns(df, tz=tz, source_tz=source_tz)


def _looks_like_time(col: "pd.Series") -> bool:
    sample = col.astype(str).head(20)
    return bool(sample.str.contains(":").all() or sample.str.fullmatch(r"\d{3,6}").all())


def load_csv(path: str | Path, start: str | None = None, end: str | None = None,
             tz: str | None = None, source_tz: str | None = None, **read_kw) -> list[Bar]:
    """Read a CSV/TXT with OHLC(V) data. English or Chinese headers, or no header at all.

    ``tz``/``source_tz``: see :func:`chan_signal_studio.data.base.normalize_columns`.
    """
    import pandas as pd

    df = read_ohlcv_csv(path, tz=tz, source_tz=source_tz, **read_kw)
    if start is not None:
        df = df[df.dt >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.dt < pd.Timestamp(end) + pd.Timedelta(days=1)]
    return bars_from_df(df)


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
