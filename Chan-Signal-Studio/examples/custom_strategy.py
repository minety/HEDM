"""Writing your own strategy: 笔级别买点 filtered by the 线段级别 direction.

A simple multi-level idea: only take stroke-level buy points while the latest
segment points up (or the last segment-level signal was a buy), and leave on
stroke-level sell points or the structural stop.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chan_signal_studio import Direction  # noqa: E402
from chan_signal_studio.backtest import Backtester  # noqa: E402
from chan_signal_studio.data import generate_bars  # noqa: E402
from chan_signal_studio.strategy import Strategy, StrategyContext  # noqa: E402


class SegFilteredStrategy(Strategy):
    def __init__(self):
        self.stop = None
        self.seg_bias = 0  # +1 after a segment-level buy point, -1 after a sell point

    def on_bar(self, ctx: StrategyContext) -> None:
        an = ctx.analyzer
        for s in ctx.new_signals:
            if s.level == "seg":
                self.seg_bias = 1 if s.is_buy else -1
        seg_up = bool(an.segs) and an.segs[-1].direction is Direction.UP
        long_ok = seg_up or self.seg_bias > 0

        holding = ctx.position > 0 or ctx.pending is not None and ctx.pending[0] > 0
        for s in ctx.new_signals:
            if s.level != "bi":
                continue
            if s.is_buy and not holding and long_ok and s.kind.value in ("B2", "B3"):
                ctx.order_target_percent(1.0, f"bi:{s.kind.value} (seg ok)")
                self.stop, holding = s.stop_price, True
            elif not s.is_buy and holding:
                ctx.close_position(f"bi:{s.kind.value}")
                self.stop, holding = None, False
        if holding and self.stop is not None and ctx.bar.close < self.stop:
            ctx.close_position("stop")
            self.stop = None


if __name__ == "__main__":
    bars = generate_bars(3000, seed=21)
    res = Backtester(SegFilteredStrategy(), rules="generic").run(bars)
    print(res.report(5))
