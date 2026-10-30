"""Reflector: turns a settled decision + outcome into one reusable lesson."""

from __future__ import annotations

import json

from ailura.agents.schemas import ReflectionLesson
from ailura.llm.base import LLMClient, LLMRequest, StructuredOutputError
from ailura.llm.structured import schema_instructions

SYSTEM = (
    "You are a trading analyst reviewing your own past decision now that the outcome is known. "
    "The outcome window may be shorter than the horizon the thesis was written for; say so if "
    "it is too short to judge. Cite the alpha figure, name the thesis component the window "
    "supports or undercuts, and give ONE concrete lesson (<= 40 words) that a future analyst "
    "can apply in a similar setup. Tag it with 2-4 short tags (e.g. 'uptrend', 'high-vol', "
    "'earnings'). A Hold is judged too: if it kept a position, judge that position; if it was "
    "flat and the stock moved more than ~3%, the Hold missed the move and is 'incorrect' (or "
    "'correct' if it avoided a drop) - never let inaction escape review."
)


class Reflector:
    role = "reflector"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def reflect(self, entry) -> ReflectionLesson:
        context = {
            "ticker": entry.ticker,
            "decision_date": entry.decision_date,
            "rating": entry.rating,
            "conviction": entry.conviction,
            "position_fraction": entry.position_fraction,
            "holding_days": entry.holding_days,
            "outcome": entry.outcome,
            "executive_summary": entry.executive_summary,
            "thesis": entry.thesis,
            "tags": entry.tags,
        }
        req = LLMRequest(
            role=self.role,
            system=f"{SYSTEM}\n\n{schema_instructions(ReflectionLesson)}",
            user="Decision and outcome (JSON):\n" + json.dumps(context, indent=1, default=str),
            schema=ReflectionLesson,
            context=context,
            tier="quick",
        )
        resp = self.llm.complete(req)
        if resp.parsed is None:
            raise StructuredOutputError(f"reflector: unparseable output {resp.text[:200]!r}")
        return resp.require(ReflectionLesson)
