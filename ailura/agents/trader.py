"""Trader: converts the research verdict into an executable proposal."""

from __future__ import annotations

from typing import Any, ClassVar

from ailura.agents.base import Agent
from ailura.agents.schemas import TraderProposal
from ailura.graph.state import TradingState


class Trader(Agent[TraderProposal]):
    role: ClassVar[str] = "trader"
    schema = TraderProposal
    system_prompt = (
        "You are the Trader. Turn the Research Manager's verdict into a concrete proposal: "
        "action (BUY/SELL/HOLD; Overweight maps to BUY, Underweight to SELL), rating, "
        "conviction, entry price, stop-loss and take-profit as ABSOLUTE prices in the quote "
        "currency (use ATR and the 60-day support/resistance levels), a signed position "
        "fraction of the maximum allocation, and a holding horizon in trading days. Respect the "
        "portfolio limits given. Stop-loss must be below entry for longs and above for shorts. "
        "HOLD keeps portfolio.current_weight unchanged; it is not an exit."
    )

    def build_context(self, state: TradingState) -> dict[str, Any]:
        f = state.features
        return {
            "ticker": state.ticker,
            "date": str(state.trade_date),
            "verdict": state.research_verdict.model_dump() if state.research_verdict else None,
            "price": {
                "close": f.get("close"),
                "atr_14": f.get("atr_14"),
                "atr_pct": f.get("atr_pct"),
                "support_60d": f.get("support_60d"),
                "resistance_60d": f.get("resistance_60d"),
                "rvol_20": f.get("rvol_20"),
                "vol_regime": f.get("vol_regime"),
                "trend_regime": f.get("trend_regime"),
            },
            "portfolio": state.portfolio.model_dump(),
            "past_lessons": state.memories,
        }
