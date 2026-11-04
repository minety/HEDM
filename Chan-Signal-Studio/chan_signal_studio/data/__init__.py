"""Pluggable data sources. AKShare and ccxt adapters import their packages lazily."""

from .base import DataSource, bars_from_df, bars_to_df, normalize_columns
from .csv_source import CSVSource, load_csv
from .resample import filter_session, resample_ohlcv
from .spy_history import build_spy_bars, download_spy_1m
from .synthetic import SyntheticSource, generate_bars


def __getattr__(name: str):
    if name == "AKShareSource":
        from .akshare_source import AKShareSource

        return AKShareSource
    if name == "CCXTSource":
        from .ccxt_source import CCXTSource

        return CCXTSource
    raise AttributeError(name)


__all__ = [
    "DataSource", "bars_from_df", "bars_to_df", "normalize_columns",
    "CSVSource", "load_csv", "SyntheticSource", "generate_bars",
    "filter_session", "resample_ohlcv", "build_spy_bars", "download_spy_1m",
    "AKShareSource", "CCXTSource",
]
