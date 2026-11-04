"""Data quality report for an OHLCV frame, optionally checked against a reference."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def quality_report(df: "pd.DataFrame", max_wick: float = 0.02) -> dict:
    """Coverage and sanity statistics for a normalized frame (dt, open, high, low, close, volume)."""
    import pandas as pd

    d = df.sort_values("dt")
    dt = pd.DatetimeIndex(d.dt)
    day = dt.normalize()
    rep: dict = {"rows": len(d), "first": dt[0], "last": dt[-1]}
    same_day = day[1:] == day[:-1]
    steps = (dt[1:] - dt[:-1])[same_day]
    rep["interval_minutes"] = float(steps.to_series().mode().iloc[0].total_seconds() / 60) if len(steps) else None
    per_day = pd.Series(1, index=day).groupby(level=0).size()
    rep["days"] = len(per_day)
    rep["bars_per_day"] = per_day.value_counts().head(5).to_dict()
    tod = pd.Series(dt.strftime("%H:%M"), index=day)
    rep["typical_first_bar"] = tod.groupby(level=0).first().mode().iloc[0]
    rep["typical_last_bar"] = tod.groupby(level=0).last().mode().iloc[0]
    rep["days_per_year"] = per_day.groupby(per_day.index.year).size().to_dict()
    days = per_day.index
    gaps = pd.Series(days[1:] - days[:-1], index=days[1:])
    long_gaps = gaps[gaps > pd.Timedelta(days=5)].sort_values(ascending=False).head(8)
    rep["long_gaps"] = [(str((end - g).date()), str(end.date()), g.days) for end, g in long_gaps.items()]
    body_hi = d[["open", "close"]].max(axis=1)
    body_lo = d[["open", "close"]].min(axis=1)
    rep["bad_wicks"] = int(((d.high > body_hi * (1 + max_wick)) | (d.low < body_lo * (1 - max_wick))).sum())
    rep["high_below_low"] = int((d.high < d.low).sum())
    rep["non_positive_prices"] = int((d[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    rep["zero_volume_share"] = float((d.volume <= 0).mean())
    return rep


def compare_to_reference(df: "pd.DataFrame", ref: "pd.DataFrame") -> dict:
    """Match bars on timestamp and compare closes (detects adjustment and timezone shifts)."""
    import numpy as np
    import pandas as pd

    a = df.set_index("dt")[["close"]]
    b = ref.set_index("dt")[["close"]]

    def joined(shift_hours: int = 0):
        x = a.copy()
        x.index = x.index - pd.Timedelta(hours=shift_hours)
        return x.join(b, how="inner", lsuffix="_x", rsuffix="_ref")

    def ret_corr(j) -> float:
        if len(j) < 10:
            return float("nan")
        return float(np.log(j.close_x).diff().corr(np.log(j.close_ref).diff()))

    j = joined()
    out: dict = {"overlap_bars": len(j), "return_correlation": ret_corr(j)}
    if not out["return_correlation"] > 0.9:
        # timestamps probably in another zone: find the whole-hour shift that lines the bars up
        scores = {h: ret_corr(joined(h)) for h in range(-12, 13) if h}
        best = max(scores, key=lambda h: -1 if np.isnan(scores[h]) else scores[h])
        out["best_hour_shift"], out["best_shift_correlation"] = best, scores[best]
        if len(j) < 10:
            return out
    ratio = j.close_x / j.close_ref
    out["median_close_ratio"] = float(ratio.median())
    out["ratio_first_year"] = float(ratio[ratio.index < ratio.index[0] + pd.Timedelta(days=365)].median())
    out["ratio_last_year"] = float(ratio[ratio.index > ratio.index[-1] - pd.Timedelta(days=365)].median())
    out["share_off_by_0.2pct"] = float((abs(ratio / ratio.rolling(500, min_periods=1).median() - 1) > 0.002).mean())
    return out


def format_report(rep: dict, cmp: dict | None = None) -> str:
    lines = [
        f"行数 {rep['rows']:,} | {rep['first']} ~ {rep['last']}",
        f"K线周期 ≈ {rep['interval_minutes']:g} 分钟 | 交易日 {rep['days']:,} | "
        f"每天常见首根 {rep['typical_first_bar']}、末根 {rep['typical_last_bar']}",
        "每天K线数(最常见): " + ", ".join(f"{k}根×{v}天" for k, v in rep["bars_per_day"].items()),
        "每年交易日: " + " ".join(f"{y}:{n}" for y, n in rep["days_per_year"].items()),
    ]
    if rep["long_gaps"]:
        lines.append("超过5天的缺口: " + "; ".join(f"{a}→{b}({n}天)" for a, b, n in rep["long_gaps"]))
    lines.append(
        f"疑似坏点(影线超过实体2%) {rep['bad_wicks']} | 高<低 {rep['high_below_low']} | "
        f"非正价格 {rep['non_positive_prices']} | 零成交量占比 {rep['zero_volume_share'] * 100:.2f}%"
    )
    if cmp:
        if cmp.get("best_shift_correlation", 0) > 0.95:
            h = cmp["best_hour_shift"]
            lines.append(f"  → 时间整体比参考数据{'晚' if h > 0 else '早'} {abs(h)} 小时才能对齐(收益相关性 {cmp['best_shift_correlation']:.3f}): "
                         "时区不同, 用 --source-tz 指定文件时区并 --tz America/New_York 转换")
        if "median_close_ratio" in cmp:
            lines.append(
                f"与参考数据(prsdro/spy)重叠 {cmp['overlap_bars']:,} 根 | 收盘价比值 中位数 {cmp['median_close_ratio']:.4f} "
                f"(首年 {cmp['ratio_first_year']:.4f}, 末年 {cmp['ratio_last_year']:.4f}) | "
                f"收益相关性 {cmp['return_correlation']:.4f} | 偏离>0.2%的K线 {cmp['share_off_by_0.2pct'] * 100:.2f}%"
            )
            if abs(cmp["ratio_first_year"] - cmp["ratio_last_year"]) > 0.01:
                lines.append("  → 早年比值明显不同于近年: 这份数据很可能做过分红/拆分复权(参考数据为不复权价格)")
        elif "best_shift_correlation" not in cmp or not cmp["best_shift_correlation"] > 0.95:
            lines.append(f"与参考数据几乎不重叠({cmp['overlap_bars']} 根), 无法比对")
    return "\n".join(lines)
