"""Vendor file formats, timezone handling and the data quality report."""

import gzip
import zipfile

import numpy as np
import pandas as pd
import pytest

from chan_signal_studio.data import load_csv
from chan_signal_studio.data.csv_source import read_ohlcv_csv
from chan_signal_studio.data.quality import compare_to_reference, format_report, quality_report

COLS = ["open", "high", "low", "close", "volume"]


@pytest.fixture(scope="module")
def ref():
    """Three sessions of 5-minute bars (one across the March DST switch)."""
    idx = []
    for d in ("2024-03-08", "2024-03-11", "2024-03-12"):
        idx += list(pd.date_range(f"{d} 09:30", f"{d} 16:00", freq="5min", inclusive="left"))
    rng = np.random.default_rng(3)
    close = 500 + np.cumsum(rng.normal(0, 0.2, len(idx)))
    return pd.DataFrame({"dt": idx, "open": close + 0.05, "high": close + 0.3, "low": close - 0.3,
                         "close": close, "volume": rng.integers(1e4, 1e5, len(idx)).astype(float)})


def same(got, ref):
    return (len(got) == len(ref) and (got.dt.values == ref.dt.values).all()
            and np.allclose(got[COLS].values, ref[COLS].values))


def test_kibot_headerless_date_time(tmp_path, ref):
    p = tmp_path / "kibot.txt"
    pd.DataFrame({"d": ref.dt.dt.strftime("%m/%d/%Y"), "t": ref.dt.dt.strftime("%H:%M"),
                  **{c: ref[c] for c in COLS}}).to_csv(p, header=False, index=False)
    assert same(read_ohlcv_csv(p), ref)


def test_firstrate_headerless_zip(tmp_path, ref):
    p = tmp_path / "fr.zip"
    body = ref.assign(dt=ref.dt.dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(header=False, index=False)
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("SPY_5min.txt", body)
    assert same(read_ohlcv_csv(p), ref)


def test_date_and_time_columns_are_combined_not_collapsed(tmp_path, ref):
    # this layout used to keep only the date, collapsing each day into one bar
    p = tmp_path / "split.csv"
    pd.DataFrame({"Date": ref.dt.dt.strftime("%Y-%m-%d"), " Time": ref.dt.dt.strftime("%H:%M"),
                  **{c.title(): ref[c] for c in COLS}}).to_csv(p, index=False)
    got = read_ohlcv_csv(p)
    assert same(got, ref)
    assert len(load_csv(p, end="2024-03-08")) == 78  # end date is inclusive for intraday bars


def test_stooq_like_int_date_time_semicolon_gz(tmp_path, ref):
    p = tmp_path / "st.csv.gz"
    df = pd.DataFrame({"Date": ref.dt.dt.strftime("%Y%m%d").astype(int),
                       "Time": ref.dt.dt.strftime("%H%M%S").astype(int), **{c: ref[c] for c in COLS}})
    with gzip.open(p, "wt") as fh:
        df.to_csv(fh, sep=";", index=False)
    assert same(read_ohlcv_csv(p), ref)


@pytest.mark.parametrize("mult", [1, 1000])
def test_epoch_seconds_and_millis_with_tz(tmp_path, ref, mult):
    utc = ref.dt.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    secs = (utc - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")
    p = tmp_path / "tv.csv"
    pd.DataFrame({"time": secs * mult, **{c: ref[c] for c in COLS}}).to_csv(p, index=False)
    assert same(read_ohlcv_csv(p, tz="America/New_York"), ref)


def test_iso_offsets_and_naive_source_tz(tmp_path, ref):
    p = tmp_path / "iso.csv"
    stamps = ref.dt.dt.tz_localize("America/New_York").dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    ref.assign(dt=stamps).rename(columns={"dt": "timestamp"}).to_csv(p, index=False)
    assert same(read_ohlcv_csv(p), ref)
    # naive Central-time file converted to Eastern (both sides of the DST switch)
    q = tmp_path / "ct.csv"
    ref.assign(dt=ref.dt - pd.Timedelta(hours=1)).to_csv(q, index=False)
    assert same(read_ohlcv_csv(q, source_tz="America/Chicago", tz="America/New_York"), ref)


def test_whitespace_delimited(tmp_path, ref):
    p = tmp_path / "ws.txt"
    lines = [f"{t:%Y-%m-%d} {t:%H:%M} {o} {h} {lo} {c} {v}" for t, o, h, lo, c, v
             in zip(ref.dt, ref.open, ref.high, ref.low, ref.close, ref.volume)]
    p.write_text("\n".join(lines))
    assert same(read_ohlcv_csv(p), ref)


def test_quality_report_and_reference_comparison(ref):
    rep = quality_report(ref)
    assert rep["interval_minutes"] == 5 and rep["days"] == 3
    assert rep["bars_per_day"] == {78: 3}
    assert rep["typical_first_bar"] == "09:30" and rep["typical_last_bar"] == "15:55"
    assert "5 分钟" in format_report(rep)

    adj = ref.copy()  # dividend-style back-adjustment: older prices scaled down smoothly
    factor = np.linspace(0.98, 1.0, len(adj))
    adj[["open", "high", "low", "close"]] = adj[["open", "high", "low", "close"]].mul(factor, axis=0)
    cmp = compare_to_reference(adj, ref)
    assert cmp["return_correlation"] > 0.9
    assert cmp["median_close_ratio"] < 1.0 and "best_hour_shift" not in cmp

    shifted = ref.assign(dt=ref.dt + pd.Timedelta(hours=5))  # e.g. UTC stamps read as local time
    cmp2 = compare_to_reference(shifted, ref)
    assert cmp2["overlap_bars"] == 3 * 18  # 14:30-15:55 still overlaps, but with the wrong bars
    assert cmp2["return_correlation"] < 0.5
    assert cmp2["best_hour_shift"] == 5 and cmp2["best_shift_correlation"] > 0.99
    assert "晚 5 小时" in format_report(rep, cmp2)
