"""Data source interface and DataFrame <-> Bar conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..models import Bar

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

# Accepted column names (lower-cased English, common Chinese, ccxt/akshare style).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "dt": ("dt", "datetime", "date", "time", "timestamp", "trade_date", "日期", "时间", "交易日期"),
    "open": ("open", "o", "开盘", "开盘价"),
    "high": ("high", "h", "最高", "最高价"),
    "low": ("low", "l", "最低", "最低价"),
    "close": ("close", "c", "收盘", "收盘价"),
    "volume": ("volume", "vol", "v", "成交量"),
}


def normalize_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """Rename recognised OHLCV columns to dt/open/high/low/close/volume."""
    import pandas as pd

    lookup = {str(c).strip().lower(): c for c in df.columns}
    rename: dict = {}
    for target, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in lookup:
                rename[lookup[a]] = target
                break
    out = df.rename(columns=rename)
    if "dt" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={out.index.name or "index": "dt"})
        else:
            raise ValueError("no date/time column found (expected one of: %s)" % ", ".join(COLUMN_ALIASES["dt"]))
    missing = [c for c in ("open", "high", "low", "close") if c not in out.columns]
    if missing:
        raise ValueError(f"missing OHLC columns: {missing}")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out = out[["dt", "open", "high", "low", "close", "volume"]].copy()
    out["dt"] = pd.to_datetime(out["dt"])
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"]).sort_values("dt")
    out = out.drop_duplicates(subset="dt", keep="last").reset_index(drop=True)
    return out


def bars_from_df(df: "pd.DataFrame") -> list[Bar]:
    """Convert a DataFrame with OHLC(V) columns (any supported naming) into bars."""
    d = normalize_columns(df)
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
