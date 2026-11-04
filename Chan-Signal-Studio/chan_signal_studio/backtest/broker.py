"""Single-instrument simulated broker with fees, lots, T+1 and price limits.

Accounting is cash based: buying spends cash, selling (including short
selling) adds cash, and equity = cash + quantity * price * multiplier. Margin
is not modelled; leverage can be expressed by target percentages above 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from ..models import Bar
from .rules import MarketRules


@dataclass
class Fill:
    dt: datetime
    bar_index: int
    qty: float  # signed: + buy, - sell
    price: float
    fee: float
    reason: str


@dataclass
class Trade:
    direction: str  # "long" | "short"
    entry_dt: datetime
    entry_index: int
    entry_price: float  # average entry price
    exit_dt: datetime
    exit_index: int
    exit_price: float  # average exit price
    qty: float  # maximum absolute size held
    pnl: float  # net of fees
    fees: float
    entry_reason: str
    exit_reason: str
    is_open: bool = False

    @property
    def return_pct(self) -> float:
        cost = self.entry_price * self.qty
        return self.pnl / cost if cost else 0.0

    @property
    def bars_held(self) -> int:
        return self.exit_index - self.entry_index


def _is_zero(x: float, scale: float = 1.0) -> bool:
    return abs(x) <= 1e-8 * max(1.0, scale)


class Broker:
    def __init__(self, rules: MarketRules, cash: float):
        self.rules = rules
        self.cash = float(cash)
        self.qty = 0.0
        self.fills: list[Fill] = []
        self.trades: list[Trade] = []
        self._open: dict | None = None
        self._last_buy_date: date | None = None
        self._bought_today = 0.0

    # ---------------------------------------------------------------- state
    def equity(self, price: float) -> float:
        return self.cash + self.qty * price * self.rules.multiplier

    def target_qty(self, pct: float, price: float) -> float:
        if price <= 0:
            return 0.0
        raw = pct * self.equity(price) / (price * self.rules.multiplier)
        if raw < 0 and not self.rules.allow_short:
            raw = 0.0
        return self.rules.round_qty(raw)

    # ------------------------------------------------------------ execution
    def execute(self, target: float, bar: Bar, prev_day_close: float | None, reason: str) -> bool:
        """Trade toward ``target`` at the bar's open. Returns True when no retry is needed."""
        r = self.rules
        if target < 0 and not r.allow_short:
            target = 0.0
        delta = r.round_qty(target - self.qty)
        if delta == 0:
            return True
        day = bar.dt.date()
        if day != self._last_buy_date:
            self._bought_today = 0.0
        if r.price_limit and prev_day_close:
            up_lim = prev_day_close * (1 + r.price_limit)
            dn_lim = prev_day_close * (1 - r.price_limit)
            if delta > 0 and bar.low >= up_lim * (1 - 1e-6):
                return False  # locked limit-up all bar: cannot buy
            if delta < 0 and bar.high <= dn_lim * (1 + 1e-6):
                return False  # locked limit-down all bar: cannot sell
        retry = False
        if delta < 0 and r.t_plus_one and self.qty > 0:
            sellable = max(self.qty - self._bought_today, 0.0)
            reducing = min(-delta, self.qty)  # part of the sell that reduces the long
            if reducing > sellable + 1e-12:
                # Sell what is allowed today, the rest on a later bar.
                delta = -r.round_qty(sellable)
                retry = True
                if delta == 0:
                    return False

        price = bar.open * ((1 + r.slippage) if delta > 0 else (1 - r.slippage))
        if delta > 0:
            delta = self._affordable(delta, price)
            if delta <= 0:
                return True  # not enough cash; drop the order
        notional = abs(delta) * price * r.multiplier
        fee = r.fee(notional, is_sell=delta < 0)
        self.cash += -delta * price * r.multiplier - fee
        self._book(delta, price, fee, bar, reason)
        self.qty = r.round_qty(self.qty + delta)
        if _is_zero(self.qty):
            self.qty = 0.0
        if delta > 0:
            self._last_buy_date = day
            self._bought_today += delta
        return not retry

    def _affordable(self, delta: float, price: float) -> float:
        """Shrink a buy so cash never goes negative (covering a short is always allowed)."""
        r = self.rules
        if self.qty < 0 and self.qty + delta <= 0:
            return delta
        unit = price * r.multiplier
        cover = max(-self.qty, 0.0)  # part of the buy that covers a short
        budget = self.cash
        # quantity beyond the cover that the remaining cash can pay for, fees included
        extra = (budget - cover * unit * (1 + r.commission_rate) - r.min_commission) / (unit * (1 + r.commission_rate))
        allowed = r.round_qty(min(delta, cover + max(extra, 0.0)))
        return max(allowed, 0.0)

    def _book(self, delta: float, price: float, fee: float, bar: Bar, reason: str) -> None:
        self.fills.append(Fill(bar.dt, bar.index, delta, price, fee, reason))
        mult = self.rules.multiplier
        remaining = delta
        # close (part of) the existing position first
        if self._open is not None and (self.qty > 0) != (delta > 0):
            o = self._open
            close_qty = min(abs(delta), abs(self.qty))
            frac = close_qty / abs(delta)
            sign = 1 if o["direction"] == "long" else -1
            o["realized"] += sign * (price - o["avg"]) * close_qty * mult
            o["fees"] += fee * frac
            o["exit_value"] += price * close_qty
            o["exit_qty"] += close_qty
            o["size"] -= close_qty
            remaining = delta - (close_qty if delta > 0 else -close_qty)
            if _is_zero(remaining, abs(delta)):
                remaining = 0.0
            fee *= 1 - frac
            if _is_zero(abs(self.qty) - close_qty, abs(self.qty)):
                self._close_trade(bar, reason)
        if remaining != 0.0:
            if self._open is None:
                self._open = {
                    "direction": "long" if remaining > 0 else "short",
                    "entry_dt": bar.dt, "entry_index": bar.index, "avg": price,
                    "size": abs(remaining), "max_size": abs(remaining), "realized": 0.0,
                    "fees": fee, "exit_value": 0.0, "exit_qty": 0.0, "entry_reason": reason,
                }
            else:  # adding to the position
                o = self._open
                new_size = o["size"] + abs(remaining)
                o["avg"] = (o["avg"] * o["size"] + price * abs(remaining)) / new_size
                o["size"] = new_size
                o["max_size"] = max(o["max_size"], new_size)
                o["fees"] += fee

    def _close_trade(self, bar: Bar, reason: str, is_open: bool = False) -> None:
        o = self._open
        if o is None:
            return
        exit_price = o["exit_value"] / o["exit_qty"] if o["exit_qty"] else bar.close
        self.trades.append(Trade(
            direction=o["direction"], entry_dt=o["entry_dt"], entry_index=o["entry_index"],
            entry_price=o["avg"], exit_dt=bar.dt, exit_index=bar.index, exit_price=exit_price,
            qty=o["max_size"], pnl=o["realized"] - o["fees"], fees=o["fees"],
            entry_reason=o["entry_reason"], exit_reason=reason, is_open=is_open,
        ))
        self._open = None

    def mark_open_trade(self, bar: Bar) -> None:
        """At the end of a run, report a still-open position as an open trade at the last close."""
        if self._open is None or self.qty == 0:
            return
        o = dict(self._open)
        sign = 1 if o["direction"] == "long" else -1
        unreal = sign * (bar.close - o["avg"]) * abs(self.qty) * self.rules.multiplier
        self.trades.append(Trade(
            direction=o["direction"], entry_dt=o["entry_dt"], entry_index=o["entry_index"],
            entry_price=o["avg"], exit_dt=bar.dt, exit_index=bar.index, exit_price=bar.close,
            qty=o["max_size"], pnl=o["realized"] + unreal - o["fees"], fees=o["fees"],
            entry_reason=o["entry_reason"], exit_reason="end of data", is_open=True,
        ))
