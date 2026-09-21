import numpy as np
import pandas as pd

from ailura.backtest import Backtester, ExecutionConfig
from ailura.backtest.metrics import annualized_return, cumulative_return, max_drawdown, sharpe


def test_metrics_basic():
    eq = pd.Series([100, 110, 99, 120.0])
    assert np.isclose(cumulative_return(eq), 0.2)
    assert np.isclose(max_drawdown(eq), 1 - 99 / 110)
    assert sharpe(pd.Series([0.0, 0.0, 0.0])) == 0.0
    assert annualized_return(pd.Series([100.0] * 253)) == 0.0


def _prices(n=50):
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(np.linspace(100, 150, n), index=idx)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0})


def test_full_weight_tracks_asset_minus_costs():
    p = _prices()
    bt = Backtester(ExecutionConfig(commission_bps=0, slippage_bps=0))
    r = bt.run(p, pd.Series(1.0, index=p.index), "bh")
    assert np.isclose(r.metrics["cumulative_return"], 1.5 - 1)
    r2 = Backtester(ExecutionConfig(commission_bps=5, slippage_bps=5)).run(p, pd.Series(1.0, index=p.index), "bh")
    assert r2.metrics["cumulative_return"] < r.metrics["cumulative_return"]
    assert r2.trades == 1


def test_zero_weight_is_flat():
    p = _prices()
    r = Backtester().run(p, pd.Series(0.0, index=p.index))
    assert r.metrics["cumulative_return"] == 0.0 and r.trades == 0


def test_weights_are_not_lookahead():
    """Weight on day t must earn day t+1's return, never day t's."""
    idx = pd.bdate_range("2024-01-01", periods=4)
    close = pd.Series([100, 120, 120, 120.0], index=idx)
    p = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0})
    w = pd.Series([0.0, 1.0, 0.0, 0.0], index=idx)  # buys after the jump
    r = Backtester(ExecutionConfig(commission_bps=0, slippage_bps=0)).run(p, w)
    assert np.isclose(r.metrics["cumulative_return"], 0.0)


def test_long_only_clips_shorts():
    p = _prices()
    r = Backtester(ExecutionConfig(allow_short=False)).run(p, pd.Series(-1.0, index=p.index))
    assert (r.weights >= 0).all()
