import pytest

from chan_signal_studio import ChanAnalyzer, ChanConfig, Direction
from helpers import WICK, zigzag_bars


def run(pivots, legs=6, **cfg):
    an = ChanAnalyzer(ChanConfig(**cfg))
    an.feed(zigzag_bars(pivots, legs))
    return an


def test_strokes_follow_the_zigzag():
    pivots = [10, 20, 14, 24, 16, 28, 18]
    an = run(pivots)
    ends = [round(b.end_price, 2) for b in an.bis]
    # the first pivot is not a fractal; every turn after it is
    assert ends == [14 - WICK, 24 + WICK, 16 - WICK, 28 + WICK]
    dirs = [b.direction for b in an.bis]
    assert dirs == [Direction.DOWN, Direction.UP, Direction.DOWN, Direction.UP]
    for a, b in zip(an.bis, an.bis[1:]):
        assert a.end_raw == b.start_raw


def test_short_wiggle_does_not_make_a_stroke():
    # a 2-bar pullback inside an up leg is too short to be a stroke
    pivots = [10, 20, 12, 16, 15, 24, 14]
    an = run(pivots, legs=[6, 6, 6, 2, 6, 6])
    highs = [b.end_price for b in an.bis if b.direction is Direction.UP]
    assert all(abs(h - (16 + WICK)) > 1e-9 for h in highs)
    assert any(abs(h - (24 + WICK)) < 1e-9 for h in highs)


@pytest.mark.parametrize("mode", ["new", "old"])
def test_legs_of_three_bars_are_too_short(mode):
    an = run([10, 20, 14, 24, 16, 28, 18], legs=3, bi_mode=mode)
    assert an.bis == []


def test_stroke_extends_when_price_makes_a_new_extreme():
    an = ChanAnalyzer()
    bars = zigzag_bars([10, 20, 14, 24, 20, 30, 22], legs=[6, 6, 6, 2, 6, 6])
    an.feed(bars)
    ups = [b for b in an.bis if b.direction is Direction.UP]
    # the 2-bar dip at 20 is ignored, so the up stroke from 14 runs to 30
    assert any(abs(b.start_price - (14 - WICK)) < 1e-9 and abs(b.end_price - (30 + WICK)) < 1e-9 for b in ups)
