"""Agent base class: renders a prompt + structured context and calls the LLM."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

from ailura.llm.base import LLMClient, LLMRequest, LLMResponse, StructuredOutputError
from ailura.llm.structured import schema_instructions

if TYPE_CHECKING:
    from ailura.graph.state import TradingState

T = TypeVar("T", bound=BaseModel)

GLOBAL_GUIDANCE = (
    "You are a member of a systematic trading research firm. Reason from the evidence provided; "
    "do not invent numbers that are not in the context. If data is missing, say so and lower "
    "confidence. Be specific, terse, and decisive. Do not hedge with 'it depends'."
)


class Agent(Generic[T]):
    role: ClassVar[str] = "agent"
    tier: ClassVar[str] = "quick"
    schema: ClassVar[type[BaseModel]]
    system_prompt: ClassVar[str] = ""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    # -- hooks subclasses implement ----------------------------------------
    def build_context(self, state: TradingState) -> dict[str, Any]:
        raise NotImplementedError

    def build_user_prompt(self, state: TradingState, context: dict[str, Any]) -> str:
        return "Context (JSON):\n" + json.dumps(context, indent=1, default=str)

    # -- execution -------------------------------------------------------------
    def make_request(self, state: TradingState) -> LLMRequest:
        context = self.build_context(state)
        system = f"{GLOBAL_GUIDANCE}\n\n{self.system_prompt}\n\n{schema_instructions(self.schema)}"
        return LLMRequest(
            role=self.role,
            system=system,
            user=self.build_user_prompt(state, context),
            schema=self.schema,
            context=context,
            tier=self.tier,
        )

    def run(self, state: TradingState) -> T:
        request = self.make_request(state)
        response: LLMResponse = self.llm.complete(request)
        state.ledger.record(request, response)
        if response.parsed is None:
            raise StructuredOutputError(f"{self.role}: no structured output; raw={response.text[:300]!r}")
        state.transcript.append({"role": self.role, "output": response.parsed.model_dump()})
        return response.parsed  # type: ignore[return-value]


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2%}"


def fmt_num(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"
