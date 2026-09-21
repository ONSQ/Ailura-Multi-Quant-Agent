"""Risk management team (aggressive / conservative / neutral) and the Portfolio Manager."""

from __future__ import annotations

from typing import Any, ClassVar

from ailura.agents.base import Agent
from ailura.agents.schemas import FinalDecision, RiskArgument
from ailura.graph.state import TradingState


class _RiskDebator(Agent[RiskArgument]):
    schema = RiskArgument
    persona: ClassVar[str] = "neutral"

    def build_context(self, state: TradingState) -> dict[str, Any]:
        f = state.features
        others = [a for a in state.risk_debate if a.persona != self.persona]
        return {
            "ticker": state.ticker,
            "date": str(state.trade_date),
            "persona": self.persona,
            "round": state.risk_rounds_run + 1,
            "proposal": state.trader_proposal.model_dump() if state.trader_proposal else None,
            "verdict": state.research_verdict.model_dump() if state.research_verdict else None,
            "reports": state.report_digest(),
            "risk": {
                "close": f.get("close"),
                "atr_pct": f.get("atr_pct"),
                "rvol_20": f.get("rvol_20"),
                "vol_regime": f.get("vol_regime"),
                "pct_from_high_60d": f.get("pct_from_high_60d"),
                "bb_width": f.get("bb_width"),
            },
            "portfolio": state.portfolio.model_dump(),
            "others_latest": [a.model_dump() for a in others[-2:]],
            "risk_history": state.risk_digest(),
        }


class AggressiveRiskAnalyst(_RiskDebator):
    role: ClassVar[str] = "aggressive_risk"
    persona = "aggressive"
    system_prompt = (
        "You are the Aggressive Risk Analyst. Champion high-reward opportunities: argue for "
        "taking or enlarging the position when the edge justifies it, rebut over-caution with "
        "data, but stay inside the portfolio limits. persona='aggressive'."
    )


class ConservativeRiskAnalyst(_RiskDebator):
    role: ClassVar[str] = "conservative_risk"
    persona = "conservative"
    system_prompt = (
        "You are the Conservative Risk Analyst. Protect capital: stress the downside, tail risk, "
        "volatility regime, crowding and thesis fragility; argue for smaller size or tighter "
        "stops when warranted, with numbers. persona='conservative'."
    )


class NeutralRiskAnalyst(_RiskDebator):
    role: ClassVar[str] = "neutral_risk"
    persona = "neutral"
    system_prompt = (
        "You are the Neutral Risk Analyst. Weigh both sides, point out where each overstates, "
        "and recommend a balanced size with volatility-aware sizing (target annual vol / "
        "realized vol, capped at the max position). persona='neutral'."
    )


class PortfolioManager(Agent[FinalDecision]):
    role: ClassVar[str] = "portfolio_manager"
    tier: ClassVar[str] = "deep"
    schema = FinalDecision
    system_prompt = (
        "You are the Portfolio Manager and final decision-maker. Read the trader's proposal and "
        "the risk debate, decide which risk arguments are stronger, and issue the FINAL decision: "
        "a 5-tier rating, action, signed conviction, signed position fraction within limits, "
        "stop-loss / take-profit as absolute prices, a 2-3 sentence executive summary, the "
        "thesis, explicit invalidation triggers, and which past lessons you applied. Disagreement "
        "is not a reason for inaction; balanced evidence is."
    )

    def build_context(self, state: TradingState) -> dict[str, Any]:
        return {
            "ticker": state.ticker,
            "date": str(state.trade_date),
            "proposal": state.trader_proposal.model_dump() if state.trader_proposal else None,
            "verdict": state.research_verdict.model_dump() if state.research_verdict else None,
            "risk_debate": state.risk_digest(),
            "portfolio": state.portfolio.model_dump(),
            "features": {
                k: state.features.get(k)
                for k in ("close", "atr_14", "atr_pct", "rvol_20", "vol_regime", "trend_regime", "support_60d", "resistance_60d")
            },
            "past_lessons": state.memories,
        }


RISK_DEBATORS: list[type[_RiskDebator]] = [AggressiveRiskAnalyst, ConservativeRiskAnalyst, NeutralRiskAnalyst]
