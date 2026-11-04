import pandas as pd

from chan_signal_studio.cli import main
from chan_signal_studio.data import bars_from_df, generate_bars, load_csv


def test_chinese_headers_and_sorting(tmp_path):
    df = pd.DataFrame({
        "日期": ["2024-01-03", "2024-01-02"], "开盘": [10.2, 10.0], "最高": [10.5, 10.3],
        "最低": [10.0, 9.8], "收盘": [10.4, 10.1], "成交量": [1000, 900],
    })
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    bars = load_csv(p)
    assert [b.dt.day for b in bars] == [2, 3]
    assert bars[0].close == 10.1 and bars[1].volume == 1000


def test_english_headers_without_volume():
    df = pd.DataFrame({"Date": ["2024-01-02"], "Open": [1], "High": [2], "Low": [0.5], "Close": [1.5]})
    b = bars_from_df(df)[0]
    assert (b.open, b.high, b.low, b.close, b.volume) == (1, 2, 0.5, 1.5, 0.0)


def test_synthetic_is_deterministic():
    assert generate_bars(100, seed=5) == generate_bars(100, seed=5)


def test_cli_end_to_end(tmp_path, capsys):
    assert main(["analyze", "--demo", "--bars", "600", "--export", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "bis.csv").exists()
    assert main(["signals", "--demo", "--bars", "600", "--last", "3"]) == 0
    html = tmp_path / "bt.html"
    assert main(["backtest", "--demo", "--bars", "800", "--market", "cn_stock",
                 "--plot", str(html), "--trades", str(tmp_path / "t.csv")]) == 0
    assert html.exists() and html.stat().st_size > 10_000
    out = capsys.readouterr().out
    assert "总收益" in out and "笔" in out
