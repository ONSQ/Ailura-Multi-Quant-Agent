"""Researcher team: bull vs bear debate, judged by the Research Manager."""

from __future__ import annotations

from typing import Any, ClassVar

from ailura.agents.base import Agent
from ailura.agents.schemas import DebateArgument, ResearchVerdict
from ailura.graph.state import TradingState


class _Researcher(Agent[DebateArgument]):
    schema = DebateArgument
    side: ClassVar[str] = "bull"

    def build_context(self, state: TradingState) -> dict[str, Any]:
        opponent = [a for a in state.debate if a.side != self.side]
        return {
            "ticker": state.ticker,
            "date": str(state.trade_date),
            "side": self.side,
            "round": state.debate_rounds_run + 1,
            "reports": state.report_digest(),
            "opponent_latest": opponent[-1].model_dump() if opponent else None,
            "debate_history": state.debate_digest(),
            "past_lessons": state.memories,
        }


class BullResearcher(_Researcher):
    role: ClassVar[str] = "bull_researcher"
    side = "bull"
    system_prompt = (
        "You are the Bull Researcher. Build the strongest evidence-based case FOR the stock: "
        "growth, competitive advantages, positive technicals and sentiment. Directly rebut the "
        "bear's latest points; concede any that are correct. Do not repeat earlier rounds; add "
        "new evidence; if you have nothing new, return an EMPTY points list (that signals "
        "convergence and ends the debate). side='bull'."
    )


class BearResearcher(_Researcher):
    role: ClassVar[str] = "bear_researcher"
    side = "bear"
    system_prompt = (
        "You are the Bear Researcher. Build the strongest evidence-based case AGAINST the "
        "stock: risks, valuation, deteriorating indicators, negative catalysts. Directly rebut "
        "the bull's latest points; concede any that are correct. Do not repeat earlier rounds; "
        "add new evidence; if you have nothing new, return an EMPTY points list (that signals "
        "convergence and ends the debate). side='bear'."
    )


class ResearchManager(Agent[ResearchVerdict]):
    role: ClassVar[str] = "research_manager"
    tier: ClassVar[str] = "deep"
    schema = ResearchVerdict
    system_prompt = (
        "You are the Research Manager judging the bull/bear debate. Decide which side argued "
        "better on the evidence and issue exactly one rating on the 5-tier scale "
        "(Buy, Overweight, Hold, Underweight, Sell) with a signed conviction in [-1, 1]. "
        "Conflict alone is never a reason to Hold: your job is to weigh it. Judge arguments by "
        "the evidence cited, not by which side sounds more confident or concedes less; a "
        "well-supported bear case deserves the same weight as a bull case. Apply the past "
        "lessons where they fit. Give a concrete plan with sizing relative to a standard "
        "position and the triggers that would change your mind."
    )

    def build_context(self, state: TradingState) -> dict[str, Any]:
        return {
            "ticker": state.ticker,
            "date": str(state.trade_date),
            "reports": state.report_digest(),
            "debate": state.debate_digest(),
            "debate_stopped_early": state.debate_stopped_early,
            "past_lessons": state.memories,
        }
