from datetime import date, timedelta

import numpy as np

from ailura.backtest import BASELINES, Backtester, ExecutionConfig
from ailura.backtest.report import results_table


def test_all_baselines_run_and_are_causal(provider):
    s, e = date(2024, 1, 1), date(2024, 3, 29)
    prices = provider.get_ohlcv("AAPL", s, e)
    warm = provider.get_ohlcv("AAPL", s - timedelta(days=400), e)
    longer = provider.get_ohlcv("AAPL", s - timedelta(days=400), e + timedelta(days=200))
    bt = Backtester(ExecutionConfig())
    results = []
    for name, strat in BASELINES.items():
        w = strat.signals(warm).loc[prices.index]
        w2 = strat.signals(longer).loc[prices.index]
        assert np.allclose(w.to_numpy(), w2.to_numpy()), f"{name} leaks future data"
        results.append(bt.run(prices, w, name))
    assert results[0].name == "B&H" and results[0].trades == 1
    table = results_table(results)
    assert "KDJ&RSI" in table and "MDD%" in table


def test_zmr_can_short_but_others_cannot(provider):
    s, e = date(2023, 1, 1), date(2024, 3, 29)
    px = provider.get_ohlcv("MSFT", s, e)
    assert (BASELINES["ZMR"].signals(px) < 0).any()
    for n in ("MACD", "SMA", "KDJ&RSI", "B&H"):
        assert (BASELINES[n].signals(px) >= 0).all()
