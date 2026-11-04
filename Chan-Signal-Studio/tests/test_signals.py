from chan_signal_studio import ChanAnalyzer, SignalKind, analyze
from chan_signal_studio.data import generate_bars
from helpers import zigzag_bars


def kinds(an, level="bi"):
    return [s.kind for s in an.signals if s.level == level]


def test_b1_on_weaker_new_low_then_b2_on_higher_low():
    # strong entry 100->60, pivot 60-70-64-69, weak new low 69->58, then 58->66->61->65
    pivots = [80, 100, 60, 70, 64, 69, 58, 66, 61, 65]
    legs = [10, 20, 6, 6, 6, 6, 6, 6, 6]
    an = ChanAnalyzer()
    an.feed(zigzag_bars(pivots, legs))
    b1 = [s for s in an.signals if s.kind is SignalKind.B1 and s.level == "bi"]
    assert len(b1) == 1
    assert abs(b1[0].price - 57.95) < 1e-9
    assert b1[0].stop_price == b1[0].price
    d = b1[0].info["divergence"]["area"]
    assert d["leave"] < d["entry"]
    b2 = [s for s in an.signals if s.kind is SignalKind.B2 and s.level == "bi"]
    assert len(b2) == 1 and abs(b2[0].price - 60.95) < 1e-9
    assert b2[0].stop_price == b1[0].price
    assert b1[0].detect_raw > b1[0].raw_index  # known only after the fractal closes


def test_no_b1_without_divergence():
    # final leg 69->20 in 6 bars has more momentum than the entry 100->60 in 20 bars
    pivots = [80, 100, 60, 70, 64, 69, 20, 30]
    legs = [10, 20, 6, 6, 6, 6, 6]
    an = ChanAnalyzer()
    an.feed(zigzag_bars(pivots, legs))
    assert SignalKind.B1 not in kinds(an)


def test_b3_when_pullback_stays_above_zg():
    pivots = [50, 40, 60, 52, 58, 51, 70, 62, 68]
    an = ChanAnalyzer()
    an.feed(zigzag_bars(pivots, 6))
    b3 = [s for s in an.signals if s.kind is SignalKind.B3 and s.level == "bi"]
    assert len(b3) == 1
    assert abs(b3[0].price - 61.95) < 1e-9
    assert abs(b3[0].stop_price - 58.05) < 1e-9  # ZG = min of the first three highs (58 + wick)


def test_s3_mirror():
    pivots = [50, 60, 40, 48, 42, 49, 30, 38, 32]
    an = ChanAnalyzer()
    an.feed(zigzag_bars(pivots, 6))
    s3 = [s for s in an.signals if s.kind is SignalKind.S3 and s.level == "bi"]
    assert len(s3) == 1 and abs(s3[0].price - 38.05) < 1e-9


def test_signals_never_use_future_bars():
    bars = generate_bars(1500, seed=3)
    full = analyze(bars)
    for k in (200, 517, 900, 1333):
        part = analyze(bars[:k])
        expect = [(s.key, s.detect_raw) for s in full.signals if s.detect_raw < k]
        got = [(s.key, s.detect_raw) for s in part.signals]
        assert got == expect


def test_signal_detection_bar_is_after_the_point(analyzed_series):
    for an in analyzed_series:
        for s in an.signals:
            assert s.detect_raw >= s.raw_index
            assert s.detect_dt >= s.dt
