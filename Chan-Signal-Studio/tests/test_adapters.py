"""Adapter tests against stubbed vendor modules (no network)."""

import sys
import types
from datetime import datetime, timezone

import pandas as pd


def test_akshare_daily_and_minute(monkeypatch):
    calls = {}
    ak = types.ModuleType("akshare")

    def stock_zh_a_hist(symbol, period, start_date, end_date, adjust):
        calls["daily"] = (symbol, period, start_date, end_date, adjust)
        return pd.DataFrame({
            "日期": ["2024-01-03", "2024-01-02"], "开盘": [10.2, 10.0], "收盘": [10.4, 10.1],
            "最高": [10.5, 10.3], "最低": [10.0, 9.8], "成交量": [1000, 900], "成交额": [1e4, 9e3],
            "振幅": [1, 1], "涨跌幅": [1, 1], "涨跌额": [0.1, 0.1], "换手率": [0.1, 0.1],
            "股票代码": ["000001", "000001"],
        })

    def stock_zh_a_hist_min_em(**kw):
        calls["minute"] = kw
        return pd.DataFrame({
            "时间": ["2024-01-02 10:00:00", "2024-01-02 10:30:00"], "开盘": [10, 10.1],
            "收盘": [10.1, 10.2], "最高": [10.2, 10.3], "最低": [9.9, 10.0],
            "成交量": [100, 200], "成交额": [1e3, 2e3], "均价": [10.05, 10.15],
        })

    ak.stock_zh_a_hist = stock_zh_a_hist
    ak.stock_zh_a_hist_min_em = stock_zh_a_hist_min_em
    monkeypatch.setitem(sys.modules, "akshare", ak)
    from chan_signal_studio.data.akshare_source import AKShareSource

    src = AKShareSource()
    bars = src.get_bars("000001", "1d", "2024-01-01", "2024-01-31")
    assert calls["daily"] == ("000001", "daily", "20240101", "20240131", "qfq")
    assert [b.dt for b in bars] == [datetime(2024, 1, 2), datetime(2024, 1, 3)]
    assert bars[0].high == 10.3 and bars[1].close == 10.4

    mbars = src.get_bars("000001", "30m", "2024-01-02", "2024-01-02")
    assert calls["minute"]["period"] == "30"
    assert calls["minute"]["start_date"] == "2024-01-02 09:30:00"
    assert calls["minute"]["end_date"] == "2024-01-02 15:00:00"
    assert mbars[1].dt == datetime(2024, 1, 2, 10, 30)


def test_ccxt_paginates_and_dedupes(monkeypatch):
    day = 86_400_000
    t0 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    data = [[t0 + i * day, 100 + i, 101 + i, 99 + i, 100.5 + i, 10] for i in range(5)]

    class FakeExchange:
        def __init__(self, cfg):
            self.cfg = cfg

        @staticmethod
        def parse8601(s):
            return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)

        def fetch_ohlcv(self, symbol, timeframe="1m", since=None, limit=None):
            rows = [r for r in data if since is None or r[0] >= since]
            return rows[:limit]

    ccxt = types.ModuleType("ccxt")
    ccxt.fakex = FakeExchange
    monkeypatch.setitem(sys.modules, "ccxt", ccxt)
    from chan_signal_studio.data.ccxt_source import CCXTSource

    src = CCXTSource("fakex", limit_per_call=2)
    bars = src.get_bars("BTC/USDT", "1d", "2024-01-01", "2024-01-04")
    assert [b.dt.day for b in bars] == [1, 2, 3, 4]
    assert bars[0].open == 100 and bars[-1].close == 103.5
