from chan_signal_studio.models import Direction
from chan_signal_studio.segment import build_segments
from helpers import make_bis


def test_case1_no_gap_ends_segment_at_the_top():
    bis = make_bis([10, 20, 15, 25, 18, 30, 22, 26, 17, 21, 12])
    segs = build_segments(bis)
    assert segs[0].direction is Direction.UP and segs[0].finished
    assert (segs[0].start_bi, segs[0].end_bi) == (0, 4)
    assert segs[0].end_price == 30
    assert segs[1].direction is Direction.DOWN and segs[1].start_bi == 5


def test_case2_gap_needs_second_sequence_confirmation():
    # gap: the peak element [28, 35] does not overlap the previous one [18, 25]
    pivots = [10, 20, 15, 25, 18, 35, 28, 32, 24, 29, 26, 30]
    segs = build_segments(make_bis(pivots))
    assert segs[0].finished and segs[0].end_price == 35

    # without the confirming bottom fractal the segment stays open
    pending = build_segments(make_bis(pivots[:9]))
    assert not pending[0].finished and pending[0].end_price == 35


def test_case2_is_cancelled_when_price_breaks_above_the_top():
    pivots = [10, 20, 15, 25, 18, 35, 28, 32, 24, 38, 30]
    segs = build_segments(make_bis(pivots))
    assert segs[0].direction is Direction.UP
    assert segs[0].end_price == 38  # the up segment went on to the new high


def test_previous_segment_extends_when_new_segment_breaks_its_start():
    # case 1 ends the up segment at 30, but the stroke 17->33 then trades above 30
    pivots = [10, 20, 15, 25, 18, 30, 22, 26, 17, 33, 25, 31, 24, 28, 20]
    segs = build_segments(make_bis(pivots))
    assert segs[0].finished and segs[0].end_price == 33
    assert segs[1].direction is Direction.DOWN and segs[1].start_price == 33


def test_segments_alternate_and_end_at_extremes(analyzed_series):
    for an in analyzed_series:
        bis, segs = an.bis, an.segs
        for a, b in zip(segs, segs[1:]):
            assert a.direction is not b.direction
            assert a.end_bi + 1 == b.start_bi
        for s in segs:
            assert s.end_bi - s.start_bi >= 2
            inside = bis[s.start_bi : s.end_bi + 1]
            assert s.high == max(b.high for b in inside)
            assert s.low == min(b.low for b in inside)


def test_incremental_segments_match_full_rebuild(analyzed_series):
    for an in analyzed_series:
        full = build_segments(an.bis)
        assert [(s.start_bi, s.end_bi, s.finished) for s in full] == \
               [(s.start_bi, s.end_bi, s.finished) for s in an.segs]
