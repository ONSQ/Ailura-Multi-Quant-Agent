"""Baseline strategies from the paper: Buy & Hold, MACD, KDJ&RSI, ZMR, SMA.

Each strategy maps a price frame to a target-weight series using only rolling
(causal) indicators, so evaluating on the full frame leaks nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from ailura.data.indicators import enrich


class Strategy(Protocol):
    name: str

    def signals(self, prices: pd.DataFrame) -> pd.Series: ...


@dataclass
class RuleStrategy:
    name: str
    rule: Callable[[pd.DataFrame], pd.Series]
    allow_short: bool = False

    def signals(self, prices: pd.DataFrame) -> pd.Series:
        w = self.rule(enrich(prices)).astype(float)
        if not self.allow_short:
            w = w.clip(lower=0.0)
        return w.fillna(0.0)


def _buy_and_hold(e: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=e.index)


def _macd(e: pd.DataFrame) -> pd.Series:
    return (e["macd"] > e["signal"]).astype(float).where(e["signal"].notna(), 0.0)


def _kdj_rsi(e: pd.DataFrame) -> pd.Series:
    """Long when K > D and RSI not overbought; exit when K < D or RSI > 80 (stateful)."""
    k, d, r = e["kdj_k"].to_numpy(), e["kdj_d"].to_numpy(), e["rsi_14"].to_numpy()
    out = np.zeros(len(e))
    pos = 0.0
    for i in range(len(e)):
        if np.isnan(k[i]) or np.isnan(d[i]) or np.isnan(r[i]):
            out[i] = 0.0
            continue
        if pos == 0.0 and k[i] > d[i] and r[i] < 70:
            pos = 1.0
        elif pos == 1.0 and (k[i] < d[i] or r[i] > 80):
            pos = 0.0
        out[i] = pos
    return pd.Series(out, index=e.index)


def _zmr(e: pd.DataFrame) -> pd.Series:
    """Z-score mean reversion: enter long below -1, exit once back above 0; short mirrors."""
    z = e["zscore_20"].to_numpy()
    out = np.zeros(len(e))
    pos = 0.0
    for i in range(len(e)):
        if np.isnan(z[i]):
            continue
        if pos == 0.0:
            pos = 1.0 if z[i] < -1.0 else -1.0 if z[i] > 1.0 else 0.0
        elif pos > 0 and z[i] >= 0:
            pos = 0.0
        elif pos < 0 and z[i] <= 0:
            pos = 0.0
        out[i] = pos
    return pd.Series(out, index=e.index)


def _sma(e: pd.DataFrame) -> pd.Series:
    return (e["close"] > e["sma_50"]).astype(float).where(e["sma_50"].notna(), 0.0)


BASELINES: dict[str, RuleStrategy] = {
    "B&H": RuleStrategy("B&H", _buy_and_hold),
    "MACD": RuleStrategy("MACD", _macd),
    "KDJ&RSI": RuleStrategy("KDJ&RSI", _kdj_rsi),
    "ZMR": RuleStrategy("ZMR", _zmr, allow_short=True),
    "SMA": RuleStrategy("SMA", _sma),
}
