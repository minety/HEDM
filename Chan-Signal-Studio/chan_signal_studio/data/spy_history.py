"""Free SPY intraday history (2000-01 .. 2026-04, 1-minute, with volume).

Source: the public GitHub repository https://github.com/prsdro/spy, file
``data/candles_1m.csv.gz`` (US/Eastern timestamps, bars labelled by their
start minute, pre/after-hours included). That repository publishes the data
"for educational and research purposes" without an open-source licence, so
Chan-Signal-Studio downloads it on demand instead of redistributing it.

Known gaps in the source: 2016-08-01 .. 2016-09-02 is missing. A small number
of 1-minute bars carry bad prints (isolated wick spikes); :func:`clean_1m`
clips wicks to ``max_wick`` beyond the bar body, the same convention the
source project uses.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from .resample import filter_session, resample_ohlcv

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

SOURCE_REPO = "https://github.com/prsdro/spy"
SOURCE_COMMIT = "36c18e584a32cd3b4f119a15ef786e3a023fdcb7"
SOURCE_URL = f"https://raw.githubusercontent.com/prsdro/spy/{SOURCE_COMMIT}/data/candles_1m.csv.gz"
SOURCE_SHA256 = "371ed38b698322bd0119f2de34a984da524145b356c4dc8578dedc4db9406dc8"
DEFAULT_CACHE = Path.home() / ".cache" / "chan-signal-studio"


def download_spy_1m(cache_dir: str | Path = DEFAULT_CACHE, verbose: bool = True) -> Path:
    """Download (once) and verify the 1-minute file. Returns the local path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"spy_1m_{SOURCE_COMMIT[:7]}.csv.gz"
    if path.exists() and _sha256(path) == SOURCE_SHA256:
        return path
    tmp = path.with_suffix(".part")
    if verbose:
        print(f"downloading SPY 1-minute history (~53 MB) from {SOURCE_REPO} ...")
    with urllib.request.urlopen(SOURCE_URL, timeout=300) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    digest = _sha256(tmp)
    if digest != SOURCE_SHA256:
        tmp.unlink(missing_ok=True)
        raise IOError(f"checksum mismatch for {SOURCE_URL}: {digest}")
    tmp.replace(path)
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_1m(raw: "pd.DataFrame", max_wick: float = 0.02) -> "pd.DataFrame":
    """Sort, de-duplicate and clip bad-print wicks of the raw 1-minute frame."""
    import pandas as pd

    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
    df.index.name = "dt"
    if max_wick:
        body_hi = df[["open", "close"]].max(axis=1)
        body_lo = df[["open", "close"]].min(axis=1)
        df["high"] = df["high"].clip(upper=body_hi * (1 + max_wick))
        df["low"] = df["low"].clip(lower=body_lo * (1 - max_wick))
    return df


def build_spy_bars(
    freq: str = "5min",
    session: str = "rth",
    start: str | None = None,
    end: str | None = None,
    cache_dir: str | Path = DEFAULT_CACHE,
    max_wick: float = 0.02,
) -> "pd.DataFrame":
    """SPY bars at ``freq`` (pandas rule, e.g. "5min", "30min", "60min").

    ``session="rth"`` keeps 09:30-16:00 ET only; ``"all"`` keeps extended hours.
    Hourly bars on RTH are aligned to 09:30 (09:30, 10:30, ...).
    """
    import pandas as pd

    df = clean_1m(pd.read_csv(download_spy_1m(cache_dir)), max_wick=max_wick)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index < pd.Timestamp(end) + pd.Timedelta(days=1)]
    if session == "rth":
        df = filter_session(df, "09:30", "16:00")
    elif session != "all":
        raise ValueError("session must be 'rth' or 'all'")
    minutes = pd.Timedelta(freq).total_seconds() / 60
    # intraday hourly bars on RTH start at 09:30 (09:30, 10:30, ...); daily bars need no offset
    offset = "30min" if session == "rth" and 60 <= minutes < 1440 and minutes % 60 == 0 else None
    out = resample_ohlcv(df, freq, offset=offset) if minutes > 1 else df
    return out.reset_index()
