"""Quickstart: analyse, list signals, backtest, chart.

Run from the project root:  python examples/quickstart.py [path/to/bars.csv]
Without a CSV it uses synthetic demo data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chan_signal_studio import ChanAnalyzer, ChanConfig  # noqa: E402
from chan_signal_studio.backtest import Backtester  # noqa: E402
from chan_signal_studio.data import generate_bars, load_csv  # noqa: E402
from chan_signal_studio.strategy import ChanStrategy, ChanStrategyParams  # noqa: E402

bars = load_csv(sys.argv[1]) if len(sys.argv) > 1 else generate_bars(2000, seed=7)

# 1) Structure analysis, one bar at a time (works the same for live data)
config = ChanConfig(bi_mode="new", divergence_methods=("area",))
an = ChanAnalyzer(config)
for bar in bars:
    for sig in an.update(bar):  # signals that became known on this bar
        pass
print(an.summary())

# 2) Backtest the reference strategy under A-share rules
params = ChanStrategyParams(levels=("bi",), buy_kinds=("B1", "B2", "B3"), sell_kinds=("S1", "S2", "S3"))
result = Backtester(ChanStrategy(params), rules="cn_stock", initial_cash=1_000_000, config=config).run(bars)
print()
print(result.report())

# 3) Interactive chart (needs plotly)
try:
    from chan_signal_studio.plot import plot_chan

    out = Path("output")
    out.mkdir(exist_ok=True)
    plot_chan(result.analyzer, result, path=str(out / "quickstart.html"), last_n=600)
    print(f"\nchart: {out / 'quickstart.html'}")
except ImportError:
    print("\n(pip install plotly to get the chart)")
