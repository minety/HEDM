"""Interactive HTML chart (optional dependency: ``pip install plotly``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .models import Move, ZhongShu

if TYPE_CHECKING:  # pragma: no cover
    from .analyzer import ChanAnalyzer
    from .backtest.engine import BacktestResult

UP_RED, DOWN_GREEN = "#d6453d", "#1f9d55"  # 红涨绿跌
UP_GREEN, DOWN_RED = "#1f9d55", "#d6453d"
BI_COLOR, SEG_COLOR = "#3b6fd8", "#8a3ffc"
BI_ZS_FILL, BI_ZS_LINE = "rgba(255,166,0,0.18)", "rgba(230,140,0,0.9)"
SEG_ZS_FILL, SEG_ZS_LINE = "rgba(138,63,252,0.10)", "rgba(138,63,252,0.7)"


def _polyline(moves: Sequence[Move], label) -> tuple[list, list]:
    if not moves:
        return [], []
    xs = [label(moves[0].start_raw)] + [label(m.end_raw) for m in moves]
    ys = [moves[0].start_price] + [m.end_price for m in moves]
    return xs, ys


def _boxes(zss: Sequence[ZhongShu], label) -> tuple[list, list]:
    xs: list = []
    ys: list = []
    for z in zss:
        x0, x1 = label(z.start_raw), label(z.end_raw)
        xs += [x0, x1, x1, x0, x0, None]
        ys += [z.zd, z.zd, z.zg, z.zg, z.zd, None]
    return xs, ys


def plot_chan(
    analyzer: "ChanAnalyzer",
    result: "BacktestResult | None" = None,
    path: str | None = None,
    title: str = "Chan-Signal-Studio",
    last_n: int | None = None,
    levels: Sequence[str] = ("bi", "seg"),
    show_macd: bool = True,
    cn_colors: bool = True,
    include_plotlyjs: str | bool = "cdn",
    max_bars: int = 20_000,
    signal_levels: Sequence[str] | None = None,
):
    """Build (and optionally save) the chart. Returns the plotly Figure.

    The last ``max_bars`` bars are embedded (pan left to see them); ``last_n``
    sets the initially visible window.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError("plotting needs plotly: pip install plotly") from e

    bars = analyzer.bars
    if not bars:
        raise ValueError("analyzer has no bars")
    start = max(0, len(bars) - max_bars) if max_bars else 0
    view = max(start, len(bars) - last_n) if last_n else start
    window = bars[start:]
    intraday = any(b.dt.hour or b.dt.minute for b in window)
    fmt = "%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d"
    labels = [b.dt.strftime(fmt) for b in window]

    def label(i: int) -> str:
        return labels[i - start]

    up_c, dn_c = (UP_RED, DOWN_GREEN) if cn_colors else (UP_GREEN, DOWN_RED)

    rows = 1 + int(show_macd) + int(result is not None)
    heights = {1: [1.0], 2: [0.72, 0.28], 3: [0.58, 0.2, 0.22]}[rows]
    titles = ["K线 · 笔 · 线段 · 中枢 · 买卖点"] + (["MACD"] if show_macd else []) + (["权益曲线"] if result is not None else [])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=heights, subplot_titles=titles)

    fig.add_trace(go.Candlestick(
        x=[label(b.index) for b in window],
        open=[b.open for b in window], high=[b.high for b in window],
        low=[b.low for b in window], close=[b.close for b in window],
        name="K线", increasing=dict(line=dict(color=up_c), fillcolor=up_c),
        decreasing=dict(line=dict(color=dn_c), fillcolor=dn_c), showlegend=False,
    ), row=1, col=1)

    def visible(moves):
        return [m for m in moves if m.start_raw >= start]

    def visible_zs(zss):
        return [z for z in zss if z.start_raw >= start]

    if "seg" in levels:
        xs, ys = _boxes(visible_zs(analyzer.seg_zs), label)
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=SEG_ZS_FILL, mode="lines",
                                     line=dict(color=SEG_ZS_LINE, width=1, dash="dot"),
                                     name="线段中枢", hoverinfo="skip"), row=1, col=1)
    if "bi" in levels:
        xs, ys = _boxes(visible_zs(analyzer.bi_zs), label)
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor=BI_ZS_FILL, mode="lines",
                                     line=dict(color=BI_ZS_LINE, width=1), name="笔中枢",
                                     hoverinfo="skip"), row=1, col=1)
        xs, ys = _polyline(visible(analyzer.bis), label)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="笔",
                                 line=dict(color=BI_COLOR, width=1.4), marker=dict(size=3)), row=1, col=1)
    if "seg" in levels:
        xs, ys = _polyline(visible(analyzer.segs), label)
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="线段",
                                     line=dict(color=SEG_COLOR, width=3)), row=1, col=1)

    sig_levels = levels if signal_levels is None else signal_levels
    sigs = [s for s in analyzer.signals if s.raw_index >= start and s.level in sig_levels]
    for is_buy in (True, False):
        group = [s for s in sigs if s.is_buy == is_buy]
        if not group:
            continue
        span = max(b.high for b in bars[view:]) - min(b.low for b in bars[view:])
        off = span * 0.025 * (-1 if is_buy else 1)
        fig.add_trace(go.Scatter(
            x=[label(s.raw_index) for s in group],
            y=[s.price + off for s in group],
            mode="markers+text",
            text=[("" if s.level == "bi" else "段") + s.kind.value for s in group],
            textposition="bottom center" if is_buy else "top center",
            textfont=dict(size=10, color=up_c if is_buy else dn_c),
            marker=dict(symbol="triangle-up" if is_buy else "triangle-down", size=11,
                        color=up_c if is_buy else dn_c, line=dict(width=1, color="white")),
            name="买点" if is_buy else "卖点",
            hovertext=[s.describe() for s in group], hoverinfo="text",
        ), row=1, col=1)

    if result is not None and result.fills:
        fills = [f for f in result.fills if f.bar_index >= start]
        fig.add_trace(go.Scatter(
            x=[label(f.bar_index) for f in fills], y=[f.price for f in fills], mode="markers",
            marker=dict(symbol="circle-open", size=9, color=["#0b6bcb" if f.qty > 0 else "#e8590c" for f in fills],
                        line=dict(width=2)),
            name="成交", hovertext=[f"{'买' if f.qty > 0 else '卖'} {abs(f.qty):g} @ {f.price:.4f} ({f.reason})" for f in fills],
            hoverinfo="text",
        ), row=1, col=1)

    row = 2
    if show_macd:
        m = analyzer.macd
        xs = [label(i) for i in range(start, len(bars))]
        hist = m.hist[start:]
        fig.add_trace(go.Bar(x=xs, y=hist, name="MACD柱", showlegend=False,
                             marker_color=[up_c if h >= 0 else dn_c for h in hist]), row=row, col=1)
        fig.add_trace(go.Scatter(x=xs, y=m.dif[start:], name="DIF", line=dict(color="#444", width=1)), row=row, col=1)
        fig.add_trace(go.Scatter(x=xs, y=m.dea[start:], name="DEA", line=dict(color="#e6a700", width=1)), row=row, col=1)
        row += 1
    if result is not None:
        eq = result.equity
        xs = [label(i) for i in range(start, len(bars))]
        e = eq["equity"].iloc[start:]
        c = eq["close"].iloc[start:]
        # buy & hold rebased to the strategy's equity at the start of the initial view
        base = e.iloc[view - start] / c.iloc[view - start]
        fig.add_trace(go.Scatter(x=xs, y=list(e), name="策略权益", line=dict(color=BI_COLOR, width=1.6)), row=row, col=1)
        fig.add_trace(go.Scatter(x=xs, y=list(c * base), name="买入持有",
                                 line=dict(color="#999", width=1, dash="dot")), row=row, col=1)

    fig.update_layout(
        title=dict(text=title, y=0.99, yanchor="top"), template="plotly_white",
        height=440 + 230 * (rows - 1), hovermode="x",
        legend=dict(orientation="h", yref="container", y=0.955, yanchor="top", x=0),
        margin=dict(l=50, r=20, t=100, b=40), xaxis_rangeslider_visible=False, bargap=0,
    )
    fig.update_xaxes(type="category", nticks=12, showgrid=False, range=[view - start - 0.5, len(bars) - start - 0.5])
    # y ranges follow the initial view (double-click a panel to autoscale after panning)
    seen = bars[view:]
    lo, hi = min(b.low for b in seen), max(b.high for b in seen)
    pad = (hi - lo) * 0.06 or abs(hi) * 0.01 or 1.0
    fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=1)
    if show_macd:
        vals = analyzer.macd.hist[view:] + analyzer.macd.dif[view:] + analyzer.macd.dea[view:]
        m_lo, m_hi = min(vals), max(vals)
        m_pad = (m_hi - m_lo) * 0.08 or 1.0
        fig.update_yaxes(range=[m_lo - m_pad, m_hi + m_pad], row=2, col=1)
    if result is not None:
        e = result.equity["equity"].iloc[view:]
        c = result.equity["close"].iloc[view:]
        bh = c / c.iloc[0] * e.iloc[0]
        e_lo, e_hi = min(e.min(), bh.min()), max(e.max(), bh.max())
        e_pad = (e_hi - e_lo) * 0.08 or 1.0
        fig.update_yaxes(range=[e_lo - e_pad, e_hi + e_pad], row=rows, col=1)
    if path:
        fig.write_html(path, include_plotlyjs=include_plotlyjs)
    return fig
