"""Tunable parameters of the Chan analysis engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChanConfig:
    # --- 笔 -----------------------------------------------------------------
    # "new": 新笔 — fractals share no merged bar and at least 3 raw bars lie
    #        strictly between the two extreme bars.
    # "old": 老笔 — at least one independent merged bar between the fractals.
    bi_mode: str = "new"

    # --- 中枢 ---------------------------------------------------------------
    # Require the move entering a pivot to come from outside its range
    # (entering down move starts above ZG, entering up move starts below ZD).
    zs_require_entry: bool = True

    # --- MACD / 背驰 ----------------------------------------------------------
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Strength measures compared between the entering and leaving move:
    #   "area"  — MACD histogram area in the move's direction
    #   "dif"   — extreme DIF value reached during the move
    #   "slope" — price change per bar
    divergence_methods: tuple[str, ...] = ("area",)
    # "any": one method diverging is enough; "all": every method must diverge.
    divergence_mode: str = "any"
    # The leaving move counts as weaker when strength_leave < ratio * strength_entry.
    divergence_ratio: float = 1.0

    # --- 买卖点 ---------------------------------------------------------------
    # Levels on which buy/sell points are detected.
    signal_levels: tuple[str, ...] = field(default_factory=lambda: ("bi", "seg"))
    # A pivot that has grown beyond this many moves has upgraded to the next
    # level (中枢升级, nine moves); first-class points are no longer measured
    # against its entering move. Set to 0 to disable.
    max_pivot_moves_type1: int = 9

    def __post_init__(self) -> None:
        if self.bi_mode not in ("new", "old"):
            raise ValueError("bi_mode must be 'new' or 'old'")
        if self.divergence_mode not in ("any", "all"):
            raise ValueError("divergence_mode must be 'any' or 'all'")
        unknown = set(self.divergence_methods) - {"area", "dif", "slope"}
        if unknown:
            raise ValueError(f"unknown divergence methods: {sorted(unknown)}")
        bad_levels = set(self.signal_levels) - {"bi", "seg"}
        if bad_levels:
            raise ValueError(f"unknown signal levels: {sorted(bad_levels)}")
