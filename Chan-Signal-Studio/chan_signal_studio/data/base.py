"""Data source interface and DataFrame <-> Bar conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..models import Bar

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

# Accepted column names (lower-cased English, common Chinese, ccxt/akshare/TradingView style).
DATETIME_ALIASES = ("dt", "datetime", "timestamp", "date_time", "date time", "时间戳", "日期时间")
DATE_ALIASES = ("date", "trade_date", "day", "日期", "交易日期")
TIME_ALIASES = ("time", "时间", "bar_time")
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "open": ("open", "o", "开盘", "开盘价"),
    "high": ("high", "h", "最高", "最高价"),
    "low": ("low", "l", "最低", "最低价"),
    "close": ("close", "c", "last", "收盘", "收盘价"),
    "volume": ("volume", "vol", "v", "成交量"),
}
KNOWN_NAMES = set(DATETIME_ALIASES + DATE_ALIASES + TIME_ALIASES) | {a for v in COLUMN_ALIASES.values() for a in v}


def _pick(lookup: dict, aliases) -> object | None:
    return next((lookup[a] for a in aliases if a in lookup), None)


def _time_of_day(s: "pd.Series") -> "pd.Series":
    """'09:30', '9:30:00', 930, 93000 -> Timedelta since midnight."""
    import pandas as pd

    txt = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    def hhmmss(x: str) -> str:
        if x.isdigit():  # 930 / 0930 -> 09:30:00, 93000 -> 09:30:00
            x = x.zfill(4) if len(x) <= 4 else x.zfill(6)
            return f"{x[:2]}:{x[2:4]}:{x[4:6] or '00'}"
        return x if x.count(":") >= 2 else x + ":00"

    return pd.to_timedelta(txt.map(hhmmss))


_OFFSET_SUFFIX = r"(?:Z|[+-]\d{2}:?\d{2})$"


def _to_datetime(s: "pd.Series", tz: str | None = None) -> "pd.Series":
    import pandas as pd

    if pd.api.types.is_numeric_dtype(s):
        mx = s.abs().max()
        if 19_000_101 <= mx <= 21_001_231:  # yyyymmdd integers
            return pd.to_datetime(s.astype("int64").astype(str), format="%Y%m%d")
        unit = "ns" if mx > 1e17 else "us" if mx > 1e14 else "ms" if mx > 1e11 else "s"
        return pd.to_datetime(s, unit=unit, utc=True)
    txt = s.astype(str).str.strip()
    if txt.str.fullmatch(r"\d{8}").all():
        return pd.to_datetime(txt, format="%Y%m%d")
    if txt.str.contains(_OFFSET_SUFFIX, regex=True).all():
        if tz:  # exact instants, converted to the requested zone later
            return pd.to_datetime(txt, format="mixed", utc=True)
        # keep each row's own local wall time (offsets may change with DST)
        return pd.to_datetime(txt.str.replace(_OFFSET_SUFFIX, "", regex=True), format="mixed")
    try:
        out = pd.to_datetime(txt, format="mixed")
    except (ValueError, TypeError):
        out = pd.to_datetime(txt, format="mixed", utc=True)  # mixed UTC offsets
    if out.dtype == object:
        out = pd.to_datetime(txt, format="mixed", utc=True)
    return out


def _localize(t: "pd.Series", tz: str | None, source_tz: str | None) -> "pd.Series":
    """Return naive wall-clock times.

    tz-aware input (or epoch numbers, which are UTC) is converted to ``tz`` when
    given, otherwise keeps its own wall time. Naive input is read as
    ``source_tz`` when given and converted to ``tz``.
    """
    if t.dt.tz is not None:
        return (t.dt.tz_convert(tz) if tz else t).dt.tz_localize(None)
    if source_tz:
        t = t.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
        return t.dt.tz_convert(tz or source_tz).dt.tz_localize(None)
    return t


def normalize_columns(df: "pd.DataFrame", tz: str | None = None, source_tz: str | None = None) -> "pd.DataFrame":
    """Rename recognised OHLCV columns to dt/open/high/low/close/volume.

    Accepts a single datetime column, separate date and time columns (Kibot,
    Stooq, many vendor exports), epoch seconds/milliseconds (TradingView,
    ccxt) or a DatetimeIndex. See :func:`_localize` for ``tz``/``source_tz``.
    """
    import pandas as pd

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": "dt"})
    lookup = {str(c).strip().lower(): c for c in df.columns}
    c_dt, c_date, c_time = (_pick(lookup, a) for a in (DATETIME_ALIASES, DATE_ALIASES, TIME_ALIASES))
    if c_dt is not None:
        when = _to_datetime(df[c_dt], tz)
    elif c_date is not None and c_time is not None:
        day = _to_datetime(df[c_date])
        if day.dt.tz is not None:
            day = day.dt.tz_localize(None)
        when = day.dt.normalize() + _time_of_day(df[c_time])
    elif c_date is not None or c_time is not None:
        when = _to_datetime(df[c_date if c_date is not None else c_time], tz)
    else:
        raise ValueError("no date/time column found (expected one of: %s)"
                         % ", ".join(DATETIME_ALIASES + DATE_ALIASES + TIME_ALIASES))
    out = pd.DataFrame({"dt": _localize(when, tz, source_tz)})
    for target, aliases in COLUMN_ALIASES.items():
        col = _pick(lookup, aliases)
        if col is None:
            if target == "volume":
                out["volume"] = 0.0
                continue
            raise ValueError(f"missing {target!r} column")
        out[target] = pd.to_numeric(df[col].values, errors="coerce")
    out = out.dropna(subset=["dt", "open", "high", "low", "close"]).sort_values("dt", kind="stable")
    out["volume"] = out["volume"].fillna(0.0)
    out = out.drop_duplicates(subset="dt", keep="last").reset_index(drop=True)
    return out


def bars_from_df(df: "pd.DataFrame", tz: str | None = None, source_tz: str | None = None) -> list[Bar]:
    """Convert a DataFrame with OHLC(V) columns (any supported naming) into bars."""
    d = normalize_columns(df, tz=tz, source_tz=source_tz)
    return [
        Bar(
            dt=row.dt.to_pydatetime(),
            open=float(row.open),
            high=float(max(row.high, row.open, row.close)),
            low=float(min(row.low, row.open, row.close)),
            close=float(row.close),
            volume=float(row.volume),
        )
        for row in d.itertuples(index=False)
    ]


def bars_to_df(bars: list[Bar]) -> "pd.DataFrame":
    import pandas as pd

    return pd.DataFrame(
        {
            "dt": [b.dt for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
    )


class DataSource(ABC):
    """Implement :meth:`get_bars` to plug in any market or vendor."""

    @abstractmethod
    def get_bars(self, symbol: str, freq: str = "1d", start: str | None = None,
                 end: str | None = None) -> list[Bar]:
        """Return bars in ascending time order."""
