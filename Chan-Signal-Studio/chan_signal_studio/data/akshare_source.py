"""A-share data via AKShare (optional dependency: ``pip install akshare``)."""

from __future__ import annotations

import importlib

from ..models import Bar
from .base import DataSource, bars_from_df

_MINUTE = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "60m": "60"}
_PERIOD = {"1d": "daily", "1w": "weekly", "1M": "monthly"}


class AKShareSource(DataSource):
    """Daily/weekly/monthly and minute bars of A-share stocks from Eastmoney.

    ``symbol`` is the 6-digit code, e.g. ``"000001"``. ``adjust`` is "qfq"
    (前复权, default), "hfq" or "" (不复权).
    """

    def __init__(self, adjust: str = "qfq"):
        try:
            importlib.import_module("akshare")
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError("AKShareSource needs akshare: pip install akshare") from e
        self.adjust = adjust

    def get_bars(self, symbol: str, freq: str = "1d", start: str | None = None,
                 end: str | None = None) -> list[Bar]:
        import akshare as ak

        if freq in _PERIOD:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=_PERIOD[freq],
                start_date=(start or "19900101").replace("-", ""),
                end_date=(end or "20991231").replace("-", ""),
                adjust=self.adjust,
            )
        elif freq in _MINUTE:
            kw = {"symbol": symbol, "period": _MINUTE[freq], "adjust": self.adjust if freq != "1m" else ""}
            if start:
                kw["start_date"] = f"{start} 09:30:00" if len(start) <= 10 else start
            if end:
                kw["end_date"] = f"{end} 15:00:00" if len(end) <= 10 else end
            df = ak.stock_zh_a_hist_min_em(**kw)
        else:
            raise ValueError(f"unsupported freq {freq!r}; use one of {sorted(_PERIOD) + sorted(_MINUTE)}")
        if df is None or df.empty:
            return []
        return bars_from_df(df)
