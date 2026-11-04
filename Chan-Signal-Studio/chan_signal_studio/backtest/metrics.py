"""Performance statistics."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .broker import Trade

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def compute_metrics(equity: "pd.Series", trades: list[Trade], initial_cash: float,
                    closes: "pd.Series | None" = None, positions: "pd.Series | None" = None) -> dict:
    eq = equity.astype(float)
    out: dict = {}
    final = float(eq.iloc[-1]) if len(eq) else initial_cash
    out["initial_cash"] = initial_cash
    out["final_equity"] = final
    out["total_return"] = final / initial_cash - 1

    years = 0.0
    if len(eq) >= 2:
        span = eq.index[-1] - eq.index[0]
        years = span.total_seconds() / (365.25 * 86400)
    out["years"] = years
    out["cagr"] = (final / initial_cash) ** (1 / years) - 1 if years > 0 and final > 0 else float("nan")

    rets = eq.pct_change().dropna()
    if len(rets) > 1 and rets.std() > 0 and years > 0:
        periods_per_year = len(eq) / years
        out["sharpe"] = float(rets.mean() / rets.std() * math.sqrt(periods_per_year))
        downside = rets[rets < 0]
        out["sortino"] = (float(rets.mean() / downside.std() * math.sqrt(periods_per_year))
                          if len(downside) > 1 and downside.std() > 0 else float("nan"))
    else:
        out["sharpe"] = out["sortino"] = float("nan")

    dd = eq / eq.cummax() - 1
    out["max_drawdown"] = float(dd.min()) if len(dd) else 0.0
    out["calmar"] = (out["cagr"] / abs(out["max_drawdown"])
                     if out["max_drawdown"] < 0 and not math.isnan(out["cagr"]) else float("nan"))

    closed = [t for t in trades if not t.is_open]
    out["trades"] = len(closed)
    out["open_trades"] = len(trades) - len(closed)
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    out["win_rate"] = len(wins) / len(closed) if closed else float("nan")
    gross_win = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    out["profit_factor"] = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else float("nan")
    out["avg_trade_return"] = sum(t.return_pct for t in closed) / len(closed) if closed else float("nan")
    out["avg_bars_held"] = sum(t.bars_held for t in closed) / len(closed) if closed else float("nan")
    out["total_fees"] = sum(t.fees for t in trades)
    if positions is not None and len(positions):
        out["exposure"] = float((positions != 0).mean())
    if closes is not None and len(closes) >= 2:
        out["buy_and_hold_return"] = float(closes.iloc[-1] / closes.iloc[0] - 1)
    return out


_LABELS = [
    ("total_return", "总收益", "pct"), ("cagr", "年化收益", "pct"), ("buy_and_hold_return", "买入持有收益", "pct"),
    ("max_drawdown", "最大回撤", "pct"), ("sharpe", "夏普比率", "num"), ("sortino", "索提诺比率", "num"),
    ("calmar", "卡玛比率", "num"), ("trades", "交易次数", "int"), ("win_rate", "胜率", "pct"),
    ("profit_factor", "盈亏比(总盈/总亏)", "num"), ("avg_trade_return", "平均每笔收益", "pct"),
    ("avg_bars_held", "平均持仓K线数", "num"), ("exposure", "持仓时间占比", "pct"),
    ("total_fees", "总费用", "money"), ("final_equity", "期末权益", "money"),
]


def format_metrics(m: dict) -> str:
    lines = []
    for key, label, kind in _LABELS:
        if key not in m:
            continue
        v = m[key]
        if isinstance(v, float) and math.isnan(v):
            s = "-"
        elif kind == "pct":
            s = f"{v * 100:.2f}%"
        elif kind == "int":
            s = f"{int(v)}"
        elif kind == "money":
            s = f"{v:,.2f}"
        else:
            s = "∞" if v == float("inf") else f"{v:.2f}"
        lines.append(f"  {label}: {s}")
    return "\n".join(lines)
