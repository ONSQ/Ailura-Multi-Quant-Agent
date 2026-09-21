"""Wrap a TradingGraph as a backtestable strategy.

The graph is invoked every ``rebalance_every`` trading days with a strict
point-in-time provider, so each decision sees only data up to that day. The
target weight is the Portfolio Manager's signed ``position_fraction`` (already
volatility-scaled) or, with ``sizing="binary"``, the paper's BUY=1 / SELL=0 /
HOLD=keep rule. Memory is settled as of each decision date, so the reflection
loop learns online without look-ahead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ailura.graph.orchestrator import TradingGraph
from ailura.graph.state import PortfolioContext, TradingState


@dataclass
class AgentStrategy:
    graph: TradingGraph
    ticker: str
    name: str = "Ailura"
    rebalance_every: int = 1
    sizing: str = "conviction"  # "conviction" | "binary"
    on_decision: Callable[[TradingState], None] | None = None
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def signals(self, prices: pd.DataFrame) -> pd.Series:
        weights = pd.Series(0.0, index=prices.index)
        current = 0.0
        for i, ts in enumerate(prices.index):
            if i % self.rebalance_every == 0:
                state = self.graph.run(
                    self.ticker,
                    ts.date(),
                    PortfolioContext(
                        current_weight=current,
                        max_position_fraction=self.graph.config.max_position_fraction,
                        allow_short=self.graph.config.allow_short,
                        target_annual_vol=self.graph.config.target_annual_vol,
                    ),
                    save_log=False,
                )
                fd = state.final_decision
                assert fd is not None
                if self.sizing == "binary":
                    current = 1.0 if fd.action == "BUY" else 0.0 if fd.action == "SELL" else current
                else:
                    current = fd.position_fraction if fd.action != "HOLD" else current if self.sizing == "conviction_hold" else 0.0
                self.decisions.append({"date": str(ts.date()), "rating": fd.rating, "action": fd.action, "conviction": fd.conviction, "weight": current, "cost": state.ledger.summary()})
                if self.on_decision:
                    self.on_decision(state)
            weights.iloc[i] = current
        return weights
