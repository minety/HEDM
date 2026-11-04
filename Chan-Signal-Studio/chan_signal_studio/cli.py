"""Command line interface: ``chan-signal-studio analyze|signals|backtest ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyzer import ChanAnalyzer
from .config import ChanConfig
from .models import Bar


def _add_source_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data source (pick one)")
    g.add_argument("--csv", help="CSV file with date/open/high/low/close[/volume] columns (English or Chinese headers)")
    g.add_argument("--demo", action="store_true", help="use synthetic demo data")
    g.add_argument("--akshare", metavar="CODE", help="A-share code via AKShare, e.g. 000001")
    g.add_argument("--ccxt", metavar="SYMBOL", help="crypto symbol via ccxt, e.g. BTC/USDT")
    g.add_argument("--exchange", default="binance", help="ccxt exchange id (default: binance)")
    g.add_argument("--spy", metavar="FREQ", help="free SPY history 2000-2026 resampled to FREQ, e.g. 5min, 30min, 1D "
                   "(downloaded once from github.com/prsdro/spy and cached)")
    g.add_argument("--session", choices=["rth", "all"], default="rth", help="--spy: regular hours only (default) or all")
    p.add_argument("--freq", default="1d", help="bar frequency for AKShare/ccxt, e.g. 1d, 30m, 1h (default 1d)")
    p.add_argument("--start", help="start date YYYY-MM-DD")
    p.add_argument("--end", help="end date YYYY-MM-DD")
    p.add_argument("--bars", type=int, default=1500, help="number of synthetic bars for --demo")
    p.add_argument("--seed", type=int, default=7, help="random seed for --demo")
    p.add_argument("--bi-mode", choices=["new", "old"], default="new", help="新笔 (new) or 老笔 (old)")
    p.add_argument("--divergence", default="area", help="comma list of: area,dif,slope (default area)")


def _load(args) -> list[Bar]:
    if args.csv:
        from .data.csv_source import load_csv

        return load_csv(args.csv, args.start, args.end)
    if args.akshare:
        from .data.akshare_source import AKShareSource

        return AKShareSource().get_bars(args.akshare, args.freq, args.start, args.end)
    if args.ccxt:
        from .data.ccxt_source import CCXTSource

        return CCXTSource(args.exchange).get_bars(args.ccxt, args.freq, args.start, args.end)
    if args.spy:
        from .data.base import bars_from_df
        from .data.spy_history import build_spy_bars

        return bars_from_df(build_spy_bars(args.spy, args.session, args.start, args.end))
    if args.demo:
        from .data.synthetic import generate_bars

        return generate_bars(args.bars, seed=args.seed, start=args.start or "2018-01-02")
    raise SystemExit("no data source: use --csv, --spy, --demo, --akshare or --ccxt")


def _config(args) -> ChanConfig:
    methods = tuple(m.strip() for m in args.divergence.split(",") if m.strip())
    return ChanConfig(bi_mode=args.bi_mode, divergence_methods=methods)


def _source_name(args) -> str:
    if args.spy:
        return f"SPY {args.spy}"
    return args.csv or args.akshare or args.ccxt or "DEMO"


def cmd_analyze(args) -> int:
    bars = _load(args)
    an = ChanAnalyzer(_config(args))
    an.feed(bars)
    print(an.summary())
    if args.export:
        out = Path(args.export)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in an.to_frames().items():
            df.to_csv(out / f"{name}.csv", index=False)
        print(f"结构已导出到 {out}/")
    if args.plot:
        from .plot import plot_chan

        plot_chan(an, path=args.plot, title=f"Chan-Signal-Studio · {_source_name(args)}", last_n=args.last)
        print(f"图表已保存: {args.plot}")
    return 0


def cmd_signals(args) -> int:
    bars = _load(args)
    an = ChanAnalyzer(_config(args))
    an.feed(bars)
    sigs = an.signals[-args.last:] if args.last else an.signals
    if not sigs:
        print("没有检测到买卖点")
        return 0
    for s in sigs:
        alive = "" if an.is_signal_alive(s) else "  (结构已改变)"
        print(s.describe() + alive)
    return 0


def cmd_backtest(args) -> int:
    from .backtest.engine import Backtester
    from .strategy.chan_strategy import ChanStrategy, ChanStrategyParams

    bars = _load(args)
    params = ChanStrategyParams(
        levels=tuple(args.levels.split(",")),
        buy_kinds=tuple(args.buy.split(",")) if args.buy else (),
        sell_kinds=tuple(args.sell.split(",")) if args.sell else (),
        position_pct=args.position,
        use_stop=not args.no_stop,
        allow_short=args.short,
        require_trend_b1=args.trend_b1,
    )
    bt = Backtester(ChanStrategy(params), rules=args.market, initial_cash=args.cash, config=_config(args))
    res = bt.run(bars)
    print(f"数据: {_source_name(args)}  {bars[0].dt:%Y-%m-%d} ~ {bars[-1].dt:%Y-%m-%d}  共 {len(bars)} 根K线")
    print(res.report(args.show_trades))
    if args.trades:
        res.trades_df().to_csv(args.trades, index=False)
        print(f"交易明细已保存: {args.trades}")
    if args.plot:
        from .plot import plot_chan

        plot_chan(res.analyzer, res, path=args.plot,
                  title=f"Chan-Signal-Studio 回测 · {_source_name(args)} · {res.rules.name}", last_n=args.last)
        print(f"图表已保存: {args.plot}")
    return 0


def cmd_fetch_spy(args) -> int:
    from .data.spy_history import SOURCE_REPO, build_spy_bars

    df = build_spy_bars(args.freq, args.session, args.start, args.end)
    out = Path(args.out or f"data/SPY_{args.freq}_{args.session}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"SPY {args.freq} ({args.session}): {len(df)} 根K线 {df.dt.iloc[0]} ~ {df.dt.iloc[-1]} -> {out}")
    print(f"数据来源: {SOURCE_REPO} (仅供研究学习使用, 未随本项目分发)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="chan-signal-studio", description="Chan-Signal-Studio 缠论分析与回测")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="笔/线段/中枢/买卖点 结构分析")
    _add_source_args(a)
    a.add_argument("--plot", metavar="HTML", help="save an interactive chart")
    a.add_argument("--last", type=int, help="only chart the last N bars")
    a.add_argument("--export", metavar="DIR", help="export structures as CSV files")
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("signals", help="list detected buy/sell points")
    _add_source_args(s)
    s.add_argument("--last", type=int, default=20, help="show the last N signals (0 = all)")
    s.set_defaults(func=cmd_signals)

    b = sub.add_parser("backtest", help="backtest the reference buy/sell-point strategy")
    _add_source_args(b)
    b.add_argument("--market", default="generic", help="rules preset: generic, cn_stock, cn_futures, crypto, us_stock")
    b.add_argument("--cash", type=float, default=1_000_000.0)
    b.add_argument("--levels", default="bi", help="signal levels to trade: bi, seg or bi,seg")
    b.add_argument("--buy", default="B1,B2,B3", help="buy kinds to enter on")
    b.add_argument("--sell", default="S1,S2,S3", help="sell kinds to exit (or short) on")
    b.add_argument("--position", type=float, default=1.0, help="position size as a fraction of equity")
    b.add_argument("--short", action="store_true", help="open shorts on sell points (if the market allows)")
    b.add_argument("--no-stop", action="store_true", help="disable structural stops")
    b.add_argument("--trend-b1", action="store_true", help="only trade B1/S1 from trend divergence")
    b.add_argument("--trades", metavar="CSV", help="save the trade list")
    b.add_argument("--show-trades", type=int, default=10)
    b.add_argument("--plot", metavar="HTML", help="save an interactive chart with equity curve")
    b.add_argument("--last", type=int, help="only chart the last N bars")
    b.set_defaults(func=cmd_backtest)

    f = sub.add_parser("fetch-spy", help="download free SPY 1-minute history and save resampled bars as CSV")
    f.add_argument("--freq", default="5min", help="pandas frequency, e.g. 1min, 5min, 30min, 60min, 1D")
    f.add_argument("--session", choices=["rth", "all"], default="rth")
    f.add_argument("--start", help="start date YYYY-MM-DD")
    f.add_argument("--end", help="end date YYYY-MM-DD")
    f.add_argument("--out", help="output CSV (default data/SPY_<freq>_<session>.csv)")
    f.set_defaults(func=cmd_fetch_spy)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
