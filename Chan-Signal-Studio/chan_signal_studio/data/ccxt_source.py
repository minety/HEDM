"""Crypto exchange data via ccxt (optional dependency: ``pip install ccxt``)."""

from __future__ import annotations

from datetime import datetime, timezone

from ..models import Bar
from .base import DataSource


class CCXTSource(DataSource):
    """OHLCV from any ccxt exchange, e.g. ``CCXTSource("binance")``.

    ``symbol`` uses ccxt notation ("BTC/USDT"); ``freq`` is a ccxt timeframe
    ("1m", "5m", "1h", "1d", ...). Times are returned as naive UTC.
    """

    def __init__(self, exchange: str = "binance", limit_per_call: int = 1000, **exchange_kw):
        try:
            import ccxt
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError("CCXTSource needs ccxt: pip install ccxt") from e
        self.exchange = getattr(ccxt, exchange)({"enableRateLimit": True, **exchange_kw})
        self.limit = limit_per_call

    def get_bars(self, symbol: str, freq: str = "1d", start: str | None = None,
                 end: str | None = None) -> list[Bar]:
        ex = self.exchange
        since = ex.parse8601(f"{start}T00:00:00Z") if start else None
        until = ex.parse8601(f"{end}T23:59:59Z") if end else None
        rows: list[list] = []
        while True:
            batch = ex.fetch_ohlcv(symbol, timeframe=freq, since=since, limit=self.limit)
            if not batch:
                break
            rows.extend(batch)
            if since is None or len(batch) < self.limit:
                break
            since = batch[-1][0] + 1
            if until is not None and since > until:
                break
        seen: set[int] = set()
        bars: list[Bar] = []
        for ts, o, h, lo, c, v in rows:
            if ts in seen or (until is not None and ts > until):
                continue
            seen.add(ts)
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).replace(tzinfo=None)
            bars.append(Bar(dt, float(o), float(max(h, o, c)), float(min(lo, o, c)), float(c), float(v or 0.0)))
        bars.sort(key=lambda b: b.dt)
        return bars
