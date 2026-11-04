"""SPY study: do Chan buy/sell points carry information on 26 years of SPY?

    python examples/spy_study.py            # full study (~2-3 minutes)
    python examples/spy_study.py --quick    # 5-minute signals only

1. Builds SPY bars (2000-01 .. 2026-04) from the free 1-minute history.
2. Forward returns after each signal type, entering at the next bar's open,
   compared with entering at any bar (SPY's own drift). Also split into two
   halves with at most one signal per day, to check stability.
3. Backtests of the reference strategy on 5-minute, 30-minute and daily bars
   against buy-and-hold, plus a "hold, step aside on segment S1" overlay.

Caveats: forward windows of clustered signals overlap, so t-values are
optimistic; many signal/horizon combinations are tested; buy-and-hold here is
price only (no dividends).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chan_signal_studio import analyze  # noqa: E402
from chan_signal_studio.backtest import Backtester, get_rules  # noqa: E402
from chan_signal_studio.data import bars_from_df, build_spy_bars  # noqa: E402
from chan_signal_studio.strategy import ChanStrategy, ChanStrategyParams, Strategy  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


class BuyAndHold(Strategy):
    def __init__(self):
        self.done = False

    def on_bar(self, ctx):
        if not self.done:
            ctx.order_target_percent(1.0, "buy&hold")
            self.done = True


class StepAsideOnSegS1(Strategy):
    """Long by default; flat after a segment-level S1 until any buy point or the S1 high is retaken."""

    def __init__(self):
        self.out_until = None
        self.started = False

    def on_bar(self, ctx):
        if not self.started:
            ctx.order_target_percent(1.0, "initial long")
            self.started = True
            return
        for s in ctx.new_signals:
            if s.level == "seg" and s.kind.value == "S1" and self.out_until is None:
                self.out_until = s.stop_price
                ctx.close_position("seg:S1 step aside")
            elif s.is_buy and self.out_until is not None:
                self.out_until = None
                ctx.order_target_percent(1.0, f"back in on {s.level}:{s.kind.value}")
        if self.out_until is not None and ctx.bar.close > self.out_until:
            self.out_until = None
            ctx.order_target_percent(1.0, "S1 high retaken")


def forward_returns(bars, an, horizons):
    op = np.array([b.open for b in bars])
    cl = np.array([b.close for b in bars])
    n = len(bars)

    def fwd(idx, h):
        idx = np.asarray([i for i in idx if i + h < n], dtype=int)
        return cl[idx + h] / op[idx + 1] - 1

    base = {k: fwd(np.arange(n - 1), h) for k, h in horizons.items()}
    sig = pd.DataFrame([(s.level, s.kind.value, s.detect_raw) for s in an.signals], columns=["level", "kind", "det"])
    rows = []
    for (lvl, kind), g in sig.groupby(["level", "kind"]):
        r = {"级别": lvl, "类型": kind, "次数": len(g)}
        sign = 1 if kind.startswith("B") else -1
        for k, h in horizons.items():
            f = fwd(g.det.values, h)
            ex = f.mean() - base[k].mean()
            r[f"{k}超额bp"] = round(ex * 1e4, 1)
            r[f"{k}胜率"] = f"{((sign * f) > 0).mean() * 100:.0f}%"
            r[f"{k}t"] = round(ex / (f.std(ddof=1) / np.sqrt(len(f))), 1) if len(f) > 2 else float("nan")
        rows.append(r)
    head = ", ".join(f"{k} {base[k].mean() * 1e4:.1f}bp (上涨概率 {(base[k] > 0).mean() * 100:.0f}%)" for k in horizons)
    return head, pd.DataFrame(rows), fwd, sig


def stability(bars, fwd, sig, h, pairs):
    yrs = np.array([b.dt.year for b in bars])
    day = np.array([b.dt.date() for b in bars])
    n = len(bars)
    lines = []
    for label, (a, b) in {"2000-2012": (2000, 2012), "2013-2026": (2013, 2026)}.items():
        idx = np.where((yrs >= a) & (yrs <= b))[0]
        base = fwd(idx[idx < n - h - 1], h).mean()
        parts = []
        for lvl, kind in pairs:
            g = sig[(sig.level == lvl) & (sig.kind == kind)]
            g = g[(yrs[g.det] >= a) & (yrs[g.det] <= b)]
            g = g.loc[~pd.Series(day[g.det]).duplicated().values]  # one per day
            f = fwd(g.det.values, h)
            if len(f) < 3:
                parts.append(f"{lvl}{kind} n={len(f)}")
                continue
            ex = f.mean() - base
            parts.append(f"{lvl}{kind} {ex * 1e4:+.0f}bp (n={len(f)}, t={ex / (f.std(ddof=1) / np.sqrt(len(f))):+.1f})")
        lines.append(f"  {label} 基准 {base * 1e4:.0f}bp | " + " | ".join(parts))
    return "\n".join(lines)


def backtests(bars, label, slips, with_overlay):
    strategies = {
        "买入持有": BuyAndHold,
        "笔级别 B123/S123": lambda: ChanStrategy(ChanStrategyParams(levels=("bi",))),
        "线段级别 B123/S123": lambda: ChanStrategy(ChanStrategyParams(levels=("seg",))),
    }
    if with_overlay:
        strategies["持有+线段S1避险"] = StepAsideOnSegS1
    rows = []
    for slip in slips:
        rules = get_rules("us_stock", slippage=slip)
        for name, make in strategies.items():
            m = Backtester(make(), rules=rules).run(bars).metrics
            rows.append({
                "周期": label, "滑点": f"{slip * 1e4:.0f}bp", "策略": name,
                "年化": f"{m['cagr'] * 100:.1f}%", "最大回撤": f"{m['max_drawdown'] * 100:.0f}%",
                "夏普": round(m["sharpe"], 2), "交易": m["trades"],
                "持仓占比": f"{m.get('exposure', 0) * 100:.0f}%",
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5-minute signal study only")
    args = ap.parse_args()

    bars5 = bars_from_df(build_spy_bars("5min"))
    print(f"SPY 5分钟: {len(bars5)} 根K线, {bars5[0].dt:%Y-%m-%d} ~ {bars5[-1].dt:%Y-%m-%d}")
    an = analyze(bars5)
    print(an.summary().split("\n最新一笔")[0])

    head, table, fwd, sig = forward_returns(bars5, an, {"1小时": 12, "1天": 78, "5天": 390})
    print("\n== 买卖点之后的前瞻收益 (次根开盘进场, 超额 = 相对任意时刻买入) ==")
    print("基准: " + head)
    print(table.to_string(index=False))
    print("\n== 稳定性: 5天超额, 前后两段, 每天最多计一个信号 ==")
    pairs = [("bi", "B1"), ("bi", "S1"), ("seg", "B1"), ("seg", "S1"), ("bi", "B3"), ("bi", "S3")]
    print(stability(bars5, fwd, sig, 390, pairs))
    if args.quick:
        return

    print("\n== 回测 (us_stock 规则, 价格收益, 不含分红) ==")
    frames = [backtests(bars5, "5min", (0.0001, 0.0005), with_overlay=True)]
    for freq in ("30min", "1D"):
        frames.append(backtests(bars_from_df(build_spy_bars(freq)), freq, (0.0001,), with_overlay=False))
    print(pd.concat(frames).to_string(index=False))


if __name__ == "__main__":
    main()
