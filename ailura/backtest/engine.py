"""Daily, transaction-cost-aware backtester driven by a target-weight series.

Weights ``w_t`` are decided with information through the close of day ``t``
and earn the return from close ``t`` to close ``t+1`` (execution at the close).
With ``execution="next_close"`` the weight is applied one day later, a
conservative choice for strategies that need time to compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ailura.backtest.metrics import summarize


@dataclass
class ExecutionConfig:
    initial_capital: float = 100_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    execution: str = "close"  # "close" | "next_close"
    rebalance_threshold: float = 0.02  # ignore weight changes smaller than this (reduces churn)
    max_weight: float = 1.0
    allow_short: bool = False


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    returns: pd.Series
    weights: pd.Series
    turnover: pd.Series
    trades: int
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "trades": self.trades, **self.metrics}


class Backtester:
    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.cfg = config or ExecutionConfig()

    def run(self, prices: pd.DataFrame, target_weights: pd.Series, name: str = "strategy") -> BacktestResult:
        cfg = self.cfg
        close = prices["close"].astype(float)
        w = target_weights.reindex(close.index).astype(float).ffill().fillna(0.0)
        lo = -cfg.max_weight if cfg.allow_short else 0.0
        w = w.clip(lower=lo, upper=cfg.max_weight)
        if cfg.execution == "next_close":
            w = w.shift(1).fillna(0.0)

        # apply rebalance threshold: hold the previous weight unless the change is material
        held = np.zeros(len(w))
        prev = 0.0
        for i, tw in enumerate(w.to_numpy()):
            if abs(tw - prev) >= cfg.rebalance_threshold or (tw == 0.0 and prev != 0.0):
                prev = tw
            held[i] = prev
        held_s = pd.Series(held, index=w.index)

        asset_ret = close.pct_change().fillna(0.0)
        turnover = held_s.diff().abs().fillna(held_s.abs())
        cost_rate = (cfg.commission_bps + cfg.slippage_bps) / 1e4
        # weight decided at t earns return t -> t+1; cost paid at t when the position changes
        strat_ret = held_s.shift(1).fillna(0.0) * asset_ret - turnover.shift(1).fillna(0.0) * cost_rate
        strat_ret.iloc[0] = 0.0
        equity = cfg.initial_capital * (1 + strat_ret).cumprod()
        trades = int((turnover > 0).sum())
        res = BacktestResult(name=name, equity=equity, returns=strat_ret, weights=held_s, turnover=turnover, trades=trades)
        res.metrics = summarize(equity, strat_ret, trades)
        return res
