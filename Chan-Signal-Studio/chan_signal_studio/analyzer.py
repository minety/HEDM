"""ChanAnalyzer: the incremental 缠论 pipeline.

bars -> 包含处理 -> 分型 -> 笔 -> 线段 -> 中枢 (笔/线段) -> 背驰 -> 买卖点

Feed bars one at a time with :meth:`ChanAnalyzer.update`. Every structure is
computed only from bars seen so far, so the analyzer can drive a live system
or a backtest without look-ahead.
"""

from __future__ import annotations

from typing import Iterable

from .bi import BiBuilder
from .config import ChanConfig
from .indicators import IncrementalMACD
from .kline import KLineProcessor
from .models import Bar, Bi, Fractal, MergedBar, Seg, Signal, ZhongShu
from .segment import build_segments
from .signals import SignalDetector
from .zhongshu import build_zhongshu


def _seg_signature(segs: list[Seg]) -> tuple:
    if not segs:
        return (0,)
    s = segs[-1]
    return (len(segs), s.start_bi, s.end_bi, s.finished, s.end_price, s.end_raw)


class ChanAnalyzer:
    def __init__(self, config: ChanConfig | None = None):
        self.config = config or ChanConfig()
        c = self.config
        self.bars: list[Bar] = []
        self._kline = KLineProcessor(self.bars)
        self.macd = IncrementalMACD(c.macd_fast, c.macd_slow, c.macd_signal)
        self._bi = BiBuilder(self.bars, self._kline.merged, c.bi_mode)
        self.segs: list[Seg] = []
        self.bi_zs: list[ZhongShu] = []
        self.seg_zs: list[ZhongShu] = []
        self.signals: list[Signal] = []
        self._known: dict[tuple[str, str, int], Signal] = {}
        self._detector = SignalDetector(c, self.macd)

    # ------------------------------------------------------------ accessors
    @property
    def merged(self) -> list[MergedBar]:
        return self._kline.merged

    @property
    def fractals(self) -> list[Fractal]:
        return self._kline.fractals

    @property
    def bis(self) -> list[Bi]:
        return self._bi.bis

    def zhongshu(self, level: str) -> list[ZhongShu]:
        return self.bi_zs if level == "bi" else self.seg_zs

    def moves(self, level: str) -> list:
        return self.bis if level == "bi" else self.segs

    def is_signal_alive(self, sig: Signal) -> bool:
        """True while the move behind the signal still ends at the signal's point."""
        moves = self.moves(sig.level)
        return sig.move_index < len(moves) and moves[sig.move_index].end_raw == sig.raw_index

    # --------------------------------------------------------------- update
    def update(self, bar: Bar) -> list[Signal]:
        """Feed one bar. Returns the signals that became known on this bar."""
        if self.bars and bar.dt < self.bars[-1].dt:
            raise ValueError(f"bars must be in time order: {bar.dt} < {self.bars[-1].dt}")
        bar = Bar(bar.dt, bar.open, bar.high, bar.low, bar.close, bar.volume, len(self.bars))
        self.bars.append(bar)
        self.macd.update(bar.close)
        fx = self._kline.update(bar)
        if fx is None:
            return []
        changed = self._bi.add(fx)
        if changed is None:
            return []
        return self._on_bi_change(changed, bar)

    def feed(self, bars: Iterable[Bar]) -> list[Signal]:
        out: list[Signal] = []
        for b in bars:
            out.extend(self.update(b))
        return out

    def _on_bi_change(self, first_changed: int, bar: Bar) -> list[Signal]:
        c = self.config
        bis = self.bis
        self.bi_zs = build_zhongshu(bis, "bi", c.zs_require_entry, self.bi_zs, first_changed)

        old_segs = self.segs
        before = _seg_signature(old_segs)
        self.segs = build_segments(bis, old_segs, first_changed)
        seg_changed = _seg_signature(self.segs) != before
        if seg_changed:
            # kept segments are the very same objects; the first new one marks the change
            n = min(len(old_segs), len(self.segs))
            first_seg = next((i for i in range(n) if self.segs[i] is not old_segs[i]), n)
            self.seg_zs = build_zhongshu(self.segs, "seg", c.zs_require_entry, self.seg_zs, first_seg)

        new: list[Signal] = []
        if "bi" in c.signal_levels:
            new += self._detector.detect("bi", bis, self.bi_zs, self._known, bar.index, bar.dt)
        if "seg" in c.signal_levels and self.segs:
            if c.seg_confirm == "stroke":
                # evaluate once the first counter stroke after the segment's extreme exists
                run = len(bis) - 1 == self.segs[-1].end_bi + 1
            else:
                run = seg_changed
            if run:
                nested = None
                if c.seg_type1_nested:
                    nested = self._detector.type1_candidate("bi", bis[: self.segs[-1].end_bi + 1], self.bi_zs)
                new += self._detector.detect("seg", self.segs, self.seg_zs, self._known, bar.index, bar.dt,
                                             nested=nested)
        for s in new:
            self._known[s.key] = s
            self.signals.append(s)
        return new

    # -------------------------------------------------------------- reports
    def summary(self) -> str:
        lines = [
            f"K线 {len(self.bars)} 根 | 包含处理后 {len(self.merged)} 根 | 分型 {len(self.fractals)} 个",
            f"笔 {len(self.bis)} | 线段 {len(self.segs)} | 笔中枢 {len(self.bi_zs)} | 线段中枢 {len(self.seg_zs)}",
            f"买卖点 {len(self.signals)} 个",
        ]
        if self.bis:
            b = self.bis[-1]
            lines.append(f"最新一笔: {b.direction.value} {b.start_price:.4f} -> {b.end_price:.4f} ({b.start_dt} ~ {b.end_dt})")
        if self.segs:
            s = self.segs[-1]
            state = "已完成" if s.finished else "进行中"
            lines.append(f"最新线段({state}): {s.direction.value} {s.start_price:.4f} -> {s.end_price:.4f}")
        if self.bi_zs:
            z = self.bi_zs[-1]
            lines.append(f"最新笔中枢: ZD {z.zd:.4f} ~ ZG {z.zg:.4f}, {z.move_count} 笔, {z.start_dt} ~ {z.end_dt}")
        recent = self.signals[-5:]
        if recent:
            lines.append("最近买卖点:")
            lines.extend("  " + s.describe() for s in recent)
        return "\n".join(lines)

    def to_frames(self) -> dict:
        """Export structures as pandas DataFrames (keys: bis, segs, bi_zs, seg_zs, signals)."""
        import pandas as pd

        def moves_df(moves):
            return pd.DataFrame([
                {
                    "index": m.index, "direction": m.direction.value,
                    "start_dt": m.start_dt, "end_dt": m.end_dt,
                    "start_price": m.start_price, "end_price": m.end_price,
                    **({"finished": m.finished, "start_bi": m.start_bi, "end_bi": m.end_bi}
                       if isinstance(m, Seg) else {}),
                }
                for m in moves
            ])

        def zs_df(zs):
            return pd.DataFrame([
                {
                    "index": z.index, "level": z.level, "start_dt": z.start_dt, "end_dt": z.end_dt,
                    "zd": z.zd, "zg": z.zg, "dd": z.dd, "gg": z.gg, "moves": z.move_count,
                    "entry": z.entry_direction.value, "finished": z.finished,
                }
                for z in zs
            ])

        sig = pd.DataFrame([
            {
                "level": s.level, "kind": s.kind.value, "dt": s.dt, "price": s.price,
                "detect_dt": s.detect_dt, "stop_price": s.stop_price, "trend": s.trend,
                "alive": self.is_signal_alive(s),
            }
            for s in self.signals
        ])
        return {
            "bis": moves_df(self.bis), "segs": moves_df(self.segs),
            "bi_zs": zs_df(self.bi_zs), "seg_zs": zs_df(self.seg_zs), "signals": sig,
        }


def analyze(bars: Iterable[Bar], config: ChanConfig | None = None) -> ChanAnalyzer:
    """Run the full pipeline over ``bars`` and return the analyzer."""
    an = ChanAnalyzer(config)
    an.feed(bars)
    return an
