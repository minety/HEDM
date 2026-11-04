from datetime import datetime, timedelta

import numpy as np
import pytest

from chan_signal_studio.backtest import Backtester, get_rules, run_backtest
from chan_signal_studio.data import generate_bars
from chan_signal_studio.models import Bar
from chan_signal_studio.strategy import ChanStrategy, Strategy


@pytest.mark.parametrize("market,short", [("generic", False), ("cn_stock", False), ("crypto", True),
                                          ("cn_futures", True), ("us_stock", True)])
def test_accounting_reconciles(market, short):
    bars = generate_bars(2000, seed=11)
    r = run_backtest(bars, ChanStrategy(allow_short=short), rules=market)
    eq = r.equity
    assert np.allclose(eq.equity, eq.cash + eq.position * eq.close * r.rules.multiplier)
    assert sum(t.pnl for t in r.trades) == pytest.approx(eq.equity.iloc[-1] - 1_000_000, abs=1e-4)
    if r.rules.lot_size:
        lots = eq.position / r.rules.lot_size
        assert np.allclose(lots, np.round(lots))
    if not short:
        assert (eq.position >= 0).all() and (eq.cash > -1e-6).all()
    assert r.metrics["trades"] > 0


class Scripted(Strategy):
    """Targets a fixed position on given bar indices."""

    def __init__(self, plan):
        self.plan = plan

    def on_bar(self, ctx):
        if ctx.index in self.plan:
            ctx.order_target_percent(self.plan[ctx.index], "plan")


def flat_bars(n, start=datetime(2024, 1, 2, 9, 30), step=timedelta(minutes=30), price=10.0):
    return [Bar(start + step * i, price, price * 1.01, price * 0.99, price) for i in range(n)]


def test_orders_fill_at_next_open_with_slippage():
    bars = generate_bars(50, seed=1)
    r = Backtester(Scripted({10: 1.0}), rules=get_rules("generic", slippage=0.001)).run(bars)
    f = r.fills[0]
    assert f.bar_index == 11
    assert f.price == pytest.approx(bars[11].open * 1.001)


def test_t_plus_one_blocks_same_day_sell():
    # 30-minute bars, 8 per day; buy on bar 1, try to sell on bar 2 (same day)
    bars = []
    t = datetime(2024, 1, 2, 9, 30)
    for d in range(3):
        for k in range(8):
            bars.append(Bar(t + timedelta(days=d, minutes=30 * k), 10, 10.1, 9.9, 10))
    r = Backtester(Scripted({0: 1.0, 1: 0.0}), rules="cn_stock").run(bars)
    buy, sell = r.fills[0], r.fills[1]
    assert buy.qty > 0 and sell.qty < 0
    assert sell.dt.date() > buy.dt.date()
    assert sell.dt == datetime(2024, 1, 3, 9, 30)  # first bar of the next day
    assert buy.qty % 100 == 0


def test_locked_limit_up_cannot_be_bought():
    bars = [Bar(datetime(2024, 1, d), 10, 10, 10, 10) for d in range(1, 4)]
    bars.append(Bar(datetime(2024, 1, 4), 11, 11, 11, 11))  # one-price limit-up day
    bars.append(Bar(datetime(2024, 1, 5), 11.5, 11.8, 11.2, 11.6))
    r = Backtester(Scripted({2: 1.0}), rules="cn_stock").run(bars)
    assert r.fills[0].dt == datetime(2024, 1, 5)


def test_fees_include_minimum_commission_and_stamp_duty():
    rules = get_rules("cn_stock")
    assert rules.fee(1000.0, is_sell=False) == 5.0
    assert rules.fee(100_000.0, is_sell=True) == pytest.approx(25.0 + 50.0)
