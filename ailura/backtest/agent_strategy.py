"""Wrap a TradingGraph as a backtestable strategy.

The graph is invoked every ``rebalance_every`` trading days with a strict
point-in-time provider, so each decision sees only data up to that day. The
target weight is the Portfolio Manager's signed ``position_fraction`` (already
volatility-scaled; HOLD keeps the current weight) or, with ``sizing="binary"``,
the paper's BUY=1 / SELL=0 / HOLD=keep rule. Memory is settled as of each decision date, so the reflection
loop learns online without look-ahead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
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
        days_held = 0
        for i, ts in enumerate(prices.index):
            if i % self.rebalance_every == 0:
                state = self.graph.run(
                    self.ticker,
                    ts.date(),
                    PortfolioContext(
                        current_weight=current,
                        days_held=days_held,
                        max_position_fraction=self.graph.config.max_position_fraction,
                        allow_short=self.graph.config.allow_short,
                        target_annual_vol=self.graph.config.target_annual_vol,
                    ),
                    save_log=False,
                )
                fd = state.final_decision
                assert fd is not None
                prev = current
                if self.sizing == "binary":
                    current = 1.0 if fd.action == "BUY" else 0.0 if fd.action == "SELL" else current
                else:
                    current = fd.position_fraction
                if current == 0.0 or prev == 0.0 or current * prev < 0:
                    days_held = 0
                self.decisions.append({"date": str(ts.date()), "rating": fd.rating, "action": fd.action, "conviction": fd.conviction, "weight": current, "gated": "[" in fd.executive_summary and ("min-hold" in fd.executive_summary or "trend-guard" in fd.executive_summary), "cost": state.ledger.summary()})
                if self.on_decision:
                    self.on_decision(state)
            weights.iloc[i] = current
            if current != 0.0:
                days_held += 1
        return weights

    def diagnostics(self) -> dict[str, Any]:
        """Action base rates and position flips: the first things to check when calls look wrong."""
        acts: dict[str, int] = {}
        for d in self.decisions:
            acts[d["action"]] = acts.get(d["action"], 0) + 1
        w = [d["weight"] for d in self.decisions]
        flips = sum(1 for a, b in pairwise(w) if (a > 0) != (b > 0) or (a < 0) != (b < 0))
        gated = sum(1 for d in self.decisions if d.get("gated"))
        return {"decisions": len(self.decisions), "actions": acts, "position_flips": flips, "rule_overrides": gated}
