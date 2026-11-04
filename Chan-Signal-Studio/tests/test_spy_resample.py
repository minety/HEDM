import gzip
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from chan_signal_studio import ChanConfig, analyze
from chan_signal_studio.cli import main
from chan_signal_studio.data import filter_session, generate_bars, resample_ohlcv
from chan_signal_studio.data import spy_history


def minute_frame(days=("2024-07-02", "2024-07-03"), start="04:00", end="20:00", seed=0):
    rng = np.random.default_rng(seed)
    idx = []
    for d in days:
        idx += list(pd.date_range(f"{d} {start}", f"{d} {end}", freq="1min", inclusive="left"))
    idx = pd.DatetimeIndex(idx)
    close = 100 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    df = pd.DataFrame({"open": close + 0.01, "high": close + 0.05, "low": close - 0.05,
                       "close": close, "volume": rng.integers(100, 1000, len(idx)).astype(float)}, index=idx)
    df.index.name = "dt"
    return df


def test_filter_session_keeps_regular_hours():
    rth = filter_session(minute_frame())
    assert rth.index.min().strftime("%H:%M") == "09:30"
    assert rth.index.max().strftime("%H:%M") == "15:59"
    assert len(rth) == 2 * 390


def test_five_minute_bars_aggregate_correctly_and_stay_within_days():
    rth = filter_session(minute_frame())
    r = resample_ohlcv(rth, "5min")
    assert len(r) == 2 * 78
    first = rth.iloc[:5]
    row = r.iloc[0]
    assert r.index[0] == pd.Timestamp("2024-07-02 09:30")
    assert row.open == first.open.iloc[0] and row.close == first.close.iloc[-1]
    assert row.high == first.high.max() and row.low == first.low.min()
    assert row.volume == first.volume.sum()
    assert r.index[78] == pd.Timestamp("2024-07-03 09:30")  # no bin across the overnight gap


def test_hourly_bars_on_rth_start_at_0930():
    r = resample_ohlcv(filter_session(minute_frame(days=("2024-07-02",))), "60min", offset="30min")
    assert [t.strftime("%H:%M") for t in r.index] == ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]


def test_clean_1m_sorts_dedupes_and_clips_bad_wicks():
    raw = pd.DataFrame({
        "timestamp": ["2024-01-02 09:31:00", "2024-01-02 09:30:00", "2024-01-02 09:31:00"],
        "open": [100.0, 100.0, 100.0], "high": [100.2, 110.0, 100.3],
        "low": [99.9, 99.8, 99.9], "close": [100.1, 100.1, 100.2], "volume": [1, 2, 3],
    })
    df = spy_history.clean_1m(raw, max_wick=0.02)
    assert list(df.index.strftime("%H:%M")) == ["09:30", "09:31"]
    assert df.iloc[1].volume == 3  # duplicate timestamp: last row wins
    assert df.iloc[0].high == pytest.approx(100.1 * 1.02)  # bad print clipped


@pytest.fixture
def fake_spy(tmp_path, monkeypatch):
    df = minute_frame(days=("2024-07-01", "2024-07-02", "2024-07-03")).reset_index()
    df = df.rename(columns={"dt": "timestamp"})
    path = tmp_path / "spy.csv.gz"
    with gzip.open(path, "wt") as fh:
        df.sample(frac=1.0, random_state=1).to_csv(fh, index=False)  # shuffled like the source
    monkeypatch.setattr(spy_history, "download_spy_1m", lambda cache_dir=None, verbose=True: path)
    return path


def test_build_spy_bars_end_to_end(fake_spy):
    df = spy_history.build_spy_bars("5min", session="rth")
    assert len(df) == 3 * 78
    assert df.dt.is_monotonic_increasing
    allh = spy_history.build_spy_bars("30min", session="all")
    assert allh.dt.iloc[0] == datetime(2024, 7, 1, 4, 0)
    daily = spy_history.build_spy_bars("1D")
    assert len(daily) == 3


def test_cli_spy_source_and_fetch(fake_spy, tmp_path, capsys):
    out = tmp_path / "spy5.csv"
    assert main(["fetch-spy", "--freq", "5min", "--out", str(out)]) == 0
    assert len(pd.read_csv(out)) == 3 * 78
    assert main(["analyze", "--spy", "5min"]) == 0
    assert "K线 234 根" in capsys.readouterr().out


def test_segment_points_wait_for_the_first_counter_stroke():
    seg_sigs = []
    for seed in (3, 6, 11):  # series that produce segment-level points
        bars = generate_bars(4000, seed=seed)
        eager = analyze(bars, ChanConfig(seg_confirm="none"))
        confirmed = analyze(bars, ChanConfig(seg_confirm="stroke"))
        n_eager = sum(s.level == "seg" and s.kind.order == 1 for s in eager.signals)
        n_conf = sum(s.level == "seg" and s.kind.order == 1 for s in confirmed.signals)
        assert n_conf <= n_eager
        seg_sigs += [s for s in confirmed.signals if s.level == "seg"]
    assert len(seg_sigs) >= 5
    for s in seg_sigs:
        # a counter stroke (>= 4 bars between extremes, +1 bar to confirm its fractal)
        # must exist before the segment point is reported
        assert s.detect_raw >= s.raw_index + 5
