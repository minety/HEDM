from chan_signal_studio.zhongshu import build_zhongshu
from helpers import make_bis


def test_pivot_range_extension_and_finish():
    # entry up 10->20, then 20->14, 14->18, 18->13 overlap in [14, 18]
    moves = make_bis([10, 20, 14, 18, 13, 22, 19, 26])
    zs = build_zhongshu(moves, "bi")
    assert len(zs) == 1
    z = zs[0]
    assert (z.start, z.end) == (1, 4)  # 13->22 still touches; 22->19 does not
    assert (z.zd, z.zg) == (14, 18)
    assert z.finished
    assert (z.dd, z.gg) == (13, 20)  # leaving move excluded from GG/DD


def test_entry_must_come_from_outside_the_range():
    # moves 1..3 overlap in [17, 19] but the entering move 17.5->20 starts inside it
    moves = make_bis([17.5, 20, 16, 19, 17, 18])
    strict = build_zhongshu(moves, "bi", require_entry=True)
    assert strict and strict[0].start == 2  # entered by 20->16 from above instead
    loose = build_zhongshu(moves, "bi", require_entry=False)
    assert loose and loose[0].start == 1


def test_unfinished_pivot_keeps_absorbing_moves():
    moves = make_bis([10, 20, 14, 18, 13, 19, 15])
    z = build_zhongshu(moves, "bi")[-1]
    assert not z.finished and z.end == len(moves) - 1
