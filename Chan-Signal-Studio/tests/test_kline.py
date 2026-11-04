from datetime import datetime, timedelta

from chan_signal_studio.kline import KLineProcessor
from chan_signal_studio.models import Bar, FxType

T0 = datetime(2024, 1, 1)


def feed(ranges):
    bars = []
    kp = KLineProcessor(bars)
    fxs = []
    for i, (h, l) in enumerate(ranges):
        b = Bar(T0 + timedelta(days=i), (h + l) / 2, h, l, (h + l) / 2, index=i)
        bars.append(b)
        fx = kp.update(b)
        if fx:
            fxs.append(fx)
    return kp, fxs


def test_inclusion_up_takes_higher_high_and_higher_low():
    kp, _ = feed([(10, 8), (12, 9), (11.5, 9.5)])
    assert len(kp.merged) == 2
    m = kp.merged[-1]
    assert (m.high, m.low) == (12, 9.5)
    assert (m.start, m.end, m.high_raw, m.low_raw) == (1, 2, 1, 2)


def test_inclusion_down_takes_lower_high_and_lower_low():
    kp, _ = feed([(12, 9), (11, 8), (10.5, 8.5)])
    m = kp.merged[-1]
    assert (m.high, m.low) == (10.5, 8)


def test_outside_bar_is_merged_in_current_direction():
    # third bar contains the second; direction is down -> lower high, lower low
    kp, _ = feed([(12, 9), (11, 8), (11.5, 7)])
    m = kp.merged[-1]
    assert (m.high, m.low) == (11, 7)


def test_top_and_bottom_fractals_are_confirmed_by_the_next_bar():
    kp, fxs = feed([(10, 8), (12, 10), (11, 9), (9, 7), (10, 8)])
    assert [f.type for f in fxs] == [FxType.TOP, FxType.BOTTOM]
    top, bottom = fxs
    assert top.price == 12 and top.raw_index == 1 and top.confirm_raw == 2
    assert bottom.price == 7 and bottom.raw_index == 3 and bottom.confirm_raw == 4
    assert top.zone_high == 12 and top.zone_low == 8
