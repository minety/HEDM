"""Market rules: fees, lot size, T+1, shorting, price limits. Swap presets per market."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketRules:
    name: str = "generic"
    commission_rate: float = 0.0003  # per side, fraction of notional
    min_commission: float = 0.0  # per fill
    sell_tax_rate: float = 0.0  # extra tax on sells (e.g. A-share stamp duty)
    slippage: float = 0.0005  # adverse fraction of price per fill
    lot_size: float = 1.0  # tradable unit; 0 means fractional quantities
    t_plus_one: bool = False  # shares bought today can only be sold next day
    allow_short: bool = False
    multiplier: float = 1.0  # contract multiplier (futures)
    price_limit: float | None = None  # daily limit vs. previous day's close, e.g. 0.10

    def round_qty(self, qty: float) -> float:
        if self.lot_size <= 0:
            return round(qty, 8)
        lots = int(abs(qty) / self.lot_size + 1e-9)
        return lots * self.lot_size * (1 if qty >= 0 else -1)

    def fee(self, notional: float, is_sell: bool) -> float:
        if notional <= 0:
            return 0.0
        fee = max(self.min_commission, notional * self.commission_rate)
        if is_sell:
            fee += notional * self.sell_tax_rate
        return fee


PRESETS: dict[str, MarketRules] = {
    "generic": MarketRules(),
    # A股: 佣金万2.5 最低5元, 印花税卖出0.05%, 100股一手, T+1, 不能融券, 主板涨跌停10%
    "cn_stock": MarketRules(
        name="cn_stock", commission_rate=0.00025, min_commission=5.0, sell_tax_rate=0.0005,
        slippage=0.001, lot_size=100, t_plus_one=True, allow_short=False, price_limit=0.10,
    ),
    # 国内期货 (example: 1 lot, multiplier 10, T+0, two-way). Set multiplier per contract.
    "cn_futures": MarketRules(
        name="cn_futures", commission_rate=0.0001, slippage=0.0003, lot_size=1,
        t_plus_one=False, allow_short=True, multiplier=10,
    ),
    # 加密货币现货/永续: 0.1% taker fee, fractional size, 24/7
    "crypto": MarketRules(
        name="crypto", commission_rate=0.001, slippage=0.0005, lot_size=0, allow_short=True,
    ),
    # 美股: no commission, whole shares, shorting allowed
    "us_stock": MarketRules(
        name="us_stock", commission_rate=0.0, slippage=0.0005, lot_size=1, allow_short=True,
    ),
}


def get_rules(rules: str | MarketRules = "generic", **overrides) -> MarketRules:
    """Look up a preset by name (or pass a MarketRules) and apply field overrides."""
    if isinstance(rules, str):
        if rules not in PRESETS:
            raise KeyError(f"unknown market preset {rules!r}; available: {sorted(PRESETS)}")
        rules = PRESETS[rules]
    return dataclasses.replace(rules, **overrides) if overrides else rules
