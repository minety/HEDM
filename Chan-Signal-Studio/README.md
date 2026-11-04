# Chan-Signal-Studio

一个缠论分析与交易框架。输入 K 线，它会算出：

**包含处理 → 分型 → 笔 → 线段 → 中枢（笔/线段两级）→ 背驰 → 三类买卖点**

在这之上还有事件驱动的回测引擎（市场规则可插拔）、交互式图表和命令行工具。核心算法不绑定任何市场：A 股、期货、加密货币、美股之间只是数据源和交易规则不同，两者都可以替换。

- **增量计算、没有未来函数**：K 线逐根输入，每个买卖点都记下它被确认的那根 K 线。回测和实盘跑的是同一套代码，测试里也验证了用前一段数据算出的结构和用全量数据算出的一致。
- **纯 Python、依赖少**：核心只需要 numpy 和 pandas，plotly / akshare / ccxt 都是可选的。
- **速度快**：分析 3000 根日线大约 0.05 秒。

```
K线 ─▶ 包含处理 ─▶ 分型 ─▶ 笔 ─┬─▶ 笔中枢 ──┐
                              └─▶ 线段 ─▶ 线段中枢 ─┤
                       MACD ───────────────────────┴─▶ 背驰 / 三类买卖点 ─▶ 策略 ─▶ 回测撮合
```

## 安装

```bash
cd Chan-Signal-Studio
pip install -e .            # 核心
pip install -e ".[plot]"    # + 交互图表 (plotly)
pip install -e ".[all]"     # + AKShare (A股) 与 ccxt (加密货币)
```

## 快速开始

### 命令行

```bash
# 结构分析（合成数据），输出图表和 CSV
chan-signal-studio analyze --demo --plot chart.html --export out/

# 用你自己的数据：CSV 列名可以是 日期/开盘/最高/最低/收盘/成交量，也可以是 date/open/high/low/close/volume
chan-signal-studio analyze --csv 000001.csv --plot chart.html --last 500

# 列出最近的买卖点
chan-signal-studio signals --csv 000001.csv --last 10

# 按 A 股规则回测
chan-signal-studio backtest --csv 000001.csv --market cn_stock --plot bt.html --trades trades.csv

# 在线数据源（需要联网和对应的可选依赖）
chan-signal-studio backtest --akshare 000001 --freq 1d --start 2020-01-01 --market cn_stock
chan-signal-studio backtest --ccxt BTC/USDT --exchange binance --freq 4h --market crypto --short
```

也可以用 `python -m chan_signal_studio ...`，效果一样。

### Python

```python
from chan_signal_studio import ChanAnalyzer, ChanConfig
from chan_signal_studio.data import load_csv
from chan_signal_studio.backtest import Backtester
from chan_signal_studio.strategy import ChanStrategy, ChanStrategyParams

bars = load_csv("000001.csv")

an = ChanAnalyzer(ChanConfig(bi_mode="new"))
for bar in bars:                       # 实盘时每来一根新 K 线调用一次 update()
    for sig in an.update(bar):         # 这根 K 线上确认的买卖点
        print(sig.describe())

print(an.summary())
frames = an.to_frames()                # DataFrame: bis / segs / bi_zs / seg_zs / signals

result = Backtester(ChanStrategy(ChanStrategyParams(levels=("bi",))), rules="cn_stock").run(bars)
print(result.report())

from chan_signal_studio.plot import plot_chan
plot_chan(result.analyzer, result, path="bt.html", last_n=600)
```

更多示例见 `examples/quickstart.py` 和 `examples/custom_strategy.py`（用线段级别方向过滤笔级别买点）。

## 规则约定

缠论在细节上有好几种流派。本项目采用下面这些约定，带参数的可以在 `ChanConfig` 里改。

| 结构 | 规则 |
|---|---|
| 包含处理 | 方向看最近两根已合并的 K 线：向上时取高高（高点取高、低点取高），向下时取低低。第一根默认向上。 |
| 分型 | 在合并后的 K 线上找顶/底分型。第三根 K 线出现即确认，之后不再修改。 |
| 笔 | `bi_mode="new"` 为新笔：两个分型不共用合并 K 线，且两个极值 K 线之间至少隔 3 根原始 K 线。`"old"` 为老笔：中间至少有 1 根独立的合并 K 线。两个分型区间在价格上不能重叠，笔的终点必须是区间内的极值。新的一笔成立前，如果有分型突破了上一笔的起点，就回退上一笔并延伸再前一笔。 |
| 线段 | 用特征序列，按线段方向做包含处理（分型的第一、第二元素之间不合并），并要求分型顶点就是线段的极值。**第一种情况**（无缺口）：分型成立，线段即结束。**第二种情况**（有缺口）：要等新线段自己的特征序列出现反向分型，或价格跌破（涨破）原线段起点才确认；如果先突破了顶点，原线段继续。如果下一段在完成前就破了自己的起点，说明上一段结束得太早，予以延伸。 |
| 中枢 | 连续三段走势有重叠：ZG = 三者高点的最小值，ZD = 三者低点的最大值。进入段必须从区间外进入（`zs_require_entry`）。之后仍触及 [ZD, ZG] 的走势都算中枢延伸；第一个不触及的走势让中枢结束。笔和线段各有自己的中枢。 |
| 背驰 | 比较进入中枢的那段和离开中枢的那段的力度。`divergence_methods` 可选 `area`（MACD 柱面积）、`dif`（DIF 极值）、`slope`（每根 K 线的价格变化），用 `divergence_mode=any/all` 组合。MACD 柱按国内习惯 = 2 ×（DIF − DEA）。 |
| 一买/一卖 | 最新一段离开最新的中枢并创新极值，方向和进入段相同，且出现背驰。`trend=True` 表示前一个中枢完全在另一侧（趋势背驰），否则是盘整背驰。超过 9 段的中枢视为已升级到上一级别（中枢升级），不再从中取一买/一卖。止损 = 该点本身。 |
| 二买/二卖 | 一买之后第一次回调且不创新低。止损 = 一买的低点。 |
| 三买/三卖 | 中枢至少 3 段，另有一段离开，回抽那段没回到 ZG 以内（卖点看 ZD）。止损 = ZG（卖点为 ZD）。 |

买卖点在笔级别（`bi`）和线段级别（`seg`）上都会检测，用 `ChanConfig.signal_levels` 选择。走势的端点一确认就检测买卖点。如果这段走势之后继续延伸，旧的买卖点照样保留在记录里（不重画），新的极值处可能再产生一个新买卖点。`ChanAnalyzer.is_signal_alive(sig)` 可以查询某个买卖点背后的结构是否还成立。

## 市场规则预设

| 预设 | 佣金 | 最低 | 印花税 | 每手 | T+1 | 做空 | 说明 |
|---|---|---|---|---|---|---|---|
| `generic` | 万3 | – | – | 1 | ✗ | ✗ | 通用 |
| `cn_stock` | 万2.5 | 5 元 | 卖出 0.05% | 100 股 | ✓ | ✗ | 涨跌停 10%，一字涨停/跌停的 K 线无法成交 |
| `cn_futures` | 万1 | – | – | 1 手 | ✗ | ✓ | 示例乘数为 10，请按合约设置 |
| `crypto` | 0.1% | – | – | 可小数 | ✗ | ✓ | 7×24 小时 |
| `us_stock` | 0 | – | – | 1 股 | ✗ | ✓ | |

可以用 `get_rules("cn_stock", lot_size=200, price_limit=0.20)` 改单个字段（比如科创板/创业板的 20% 涨跌幅），也可以自己构造 `MarketRules(...)`。订单在**下一根 K 线的开盘价**成交，并计入滑点。不模拟保证金；需要杠杆时把仓位比例设成大于 1。

## 编写策略

```python
from chan_signal_studio.strategy import Strategy

class MyStrategy(Strategy):
    def on_bar(self, ctx):
        # ctx.bar / ctx.analyzer（bis, segs, bi_zs, seg_zs, signals, macd）/ ctx.new_signals
        # ctx.position / ctx.equity / ctx.pending
        for s in ctx.new_signals:
            if s.level == "bi" and s.kind.value == "B3":
                ctx.order_target_percent(0.5, "B3")
```

## 数据源

- `load_csv(path)` / `CSVSource(root)`：自动识别中英文列名。
- `bars_from_df(df)`：从任意 DataFrame 构造 K 线。
- `AKShareSource()`：A 股日/周/月线，以及 1/5/15/30/60 分钟线（东方财富，默认前复权）。
- `CCXTSource("binance")`：ccxt 支持的任意交易所的 K 线，自动分页。
- `generate_bars()`：分段切换趋势和震荡的合成数据，用于演示和测试。
- 想接入其他数据源，继承 `DataSource` 并实现 `get_bars()` 即可。

## 测试

```bash
pip install -e ".[dev]"
pytest
```

测试覆盖：包含处理与分型、笔的规则、线段的两种情况和线段延伸、中枢构建、各类买卖点、没有未来函数、回测账目核对（5 种市场预设下精确到分）、T+1、一字涨跌停、手续费、CSV/AKShare/ccxt 适配器（用桩替代真实接口）以及命令行。另外在 20 组随机序列上检查了结构上的不变量（线段方向交替、端点是极值、增量构建和全量重建结果一致）。

## 路线图

- 多级别联立（区间套）：同时分析多个周期，用低级别的背驰确认高级别的买卖点。
- 更多背驰判定方式和中枢算法（如段内中枢），以及参数扫描。
- 实盘/模拟盘接口，买卖点消息推送。

## 免责声明

本项目仅供研究和学习，不构成投资建议。回测结果（尤其是在合成数据上的结果）不代表未来收益。
