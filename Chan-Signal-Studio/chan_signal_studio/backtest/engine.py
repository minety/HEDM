"""Event-driven backtester: one bar at a time, orders fill at the next open."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..analyzer import ChanAnalyzer
from ..config import ChanConfig
from ..models import Bar
from ..strategy.base import Strategy, StrategyContext
from .broker import Broker, Fill, Trade
from .metrics import compute_metrics, format_metrics
from .rules import MarketRules, get_rules


@dataclass
class BacktestResult:
    equity: "object"  # pandas DataFrame indexed by dt: equity, cash, position, close
    trades: list[Trade]
    fills: list[Fill]
    metrics: dict
    analyzer: ChanAnalyzer
    rules: MarketRules
    params: dict = field(default_factory=dict)

    def trades_df(self):
        import pandas as pd

        return pd.DataFrame([
            {
                "direction": t.direction, "entry_dt": t.entry_dt, "entry_price": t.entry_price,
                "exit_dt": t.exit_dt, "exit_price": t.exit_price, "qty": t.qty, "pnl": t.pnl,
                "return_pct": t.return_pct, "bars_held": t.bars_held, "fees": t.fees,
                "entry_reason": t.entry_reason, "exit_reason": t.exit_reason, "open": t.is_open,
            }
            for t in self.trades
        ])

    def report(self, max_trades: int = 10) -> str:
        lines = [f"市场规则: {self.rules.name}", "绩效:", format_metrics(self.metrics)]
        if self.trades:
            lines.append(f"最近 {min(max_trades, len(self.trades))} 笔交易:")
            for t in self.trades[-max_trades:]:
                flag = " (未平仓)" if t.is_open else ""
                lines.append(
                    f"  {t.direction:<5} {t.entry_dt:%Y-%m-%d %H:%M} @ {t.entry_price:.4f} -> "
                    f"{t.exit_dt:%Y-%m-%d %H:%M} @ {t.exit_price:.4f}  "
                    f"{t.return_pct * 100:+.2f}%  [{t.entry_reason} / {t.exit_reason}]{flag}"
                )
        return "\n".join(lines)


class Backtester:
    def __init__(
        self,
        strategy: Strategy,
        rules: str | MarketRules = "generic",
        initial_cash: float = 1_000_000.0,
        config: ChanConfig | None = None,
    ):
        self.strategy = strategy
        self.rules = get_rules(rules)
        self.initial_cash = initial_cash
        self.config = config or ChanConfig()

    def run(self, bars: Sequence[Bar]) -> BacktestResult:
        import pandas as pd

        analyzer = ChanAnalyzer(self.config)
        broker = Broker(self.rules, self.initial_cash)
        self.strategy.on_start(analyzer)

        pending: tuple[float, str] | None = None
        prev_day_close: float | None = None
        last_close: float | None = None
        cur_day = None
        rows = []
        for bar in bars:
            day = bar.dt.date()
            if cur_day is not None and day != cur_day:
                prev_day_close = last_close
            cur_day = day

            idx = len(analyzer.bars)
            if pending is not None:
                pct, reason = pending
                # Fill at this bar's open. bar.index is set when the analyzer sees it,
                # so pass a copy with the index it will get.
                fill_bar = Bar(bar.dt, bar.open, bar.high, bar.low, bar.close, bar.volume, idx)
                target = broker.target_qty(pct, bar.open)
                if broker.execute(target, fill_bar, prev_day_close, reason):
                    pending = None

            new = analyzer.update(bar)
            ctx = StrategyContext(analyzer.bars[-1], idx, analyzer, broker, new, pending=pending)
            self.strategy.on_bar(ctx)
            if ctx._order is not None:
                pending = ctx._order

            rows.append((bar.dt, broker.equity(bar.close), broker.cash, broker.qty, bar.close))
            last_close = bar.close

        if analyzer.bars:
            broker.mark_open_trade(analyzer.bars[-1])
        eq = pd.DataFrame(rows, columns=["dt", "equity", "cash", "position", "close"]).set_index("dt")
        metrics = compute_metrics(eq["equity"], broker.trades, self.initial_cash,
                                  closes=eq["close"], positions=eq["position"])
        params = dict(vars(self.strategy.p)) if hasattr(self.strategy, "p") else {}
        return BacktestResult(eq, broker.trades, broker.fills, metrics, analyzer, self.rules, params)


def run_backtest(bars: Sequence[Bar], strategy: Strategy | None = None, rules: str | MarketRules = "generic",
                 initial_cash: float = 1_000_000.0, config: ChanConfig | None = None) -> BacktestResult:
    from ..strategy.chan_strategy import ChanStrategy

    return Backtester(strategy or ChanStrategy(), rules, initial_cash, config).run(bars)
