import numpy as np
import pandas as pd

from ailura.data import indicators as ind


def _df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6}, index=idx)


def test_sma_matches_pandas():
    df = _df()
    assert np.allclose(ind.sma(df["close"], 10).dropna(), df["close"].rolling(10).mean().dropna())


def test_rsi_bounds_and_monotone_series():
    df = _df()
    r = ind.rsi(df["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(up).dropna().iloc[-1] == 100.0


def test_macd_hist_is_line_minus_signal():
    df = _df()
    m = ind.macd(df["close"]).dropna()
    assert np.allclose(m["hist"], m["macd"] - m["signal"])


def test_bollinger_ordering_and_atr_positive():
    df = _df()
    b = ind.bollinger(df["close"]).dropna()
    assert (b["bb_upper"] >= b["bb_mid"]).all() and (b["bb_mid"] >= b["bb_lower"]).all()
    assert (ind.atr(df).dropna() > 0).all()


def test_kdj_in_range():
    k = ind.kdj(_df()).dropna()
    assert ((k["k"] >= 0) & (k["k"] <= 100)).all()


def test_snapshot_has_regime_and_score():
    s = ind.snapshot(_df())
    assert s["trend_regime"] in {"uptrend", "downtrend"}
    assert -1 <= s["technical_score"] <= 1
    assert s["support_60d"] <= s["close"] <= s["resistance_60d"] * 1.0001


def test_indicators_are_causal():
    """Changing future bars must not change today's indicator values."""
    df = _df()
    a = ind.enrich(df.iloc[:200]).iloc[-1]
    b = ind.enrich(df).iloc[199]
    for col in ["sma_50", "rsi_14", "macd", "atr_14", "kdj_k", "zscore_20"]:
        assert np.isclose(a[col], b[col]), col
