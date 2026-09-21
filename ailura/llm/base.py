"""Provider-agnostic LLM interface.

Every agent builds an :class:`LLMRequest` containing both a rendered prompt
(for real language models) and a structured ``context`` dict (the same facts,
machine-readable). Real providers use the prompt; the offline
:class:`~ailura.llm.heuristic.HeuristicLLM` uses the context. This keeps the
whole pipeline runnable and testable with zero network access.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

Tier = str  # "quick" | "deep"


@dataclass
class LLMRequest:
    role: str
    system: str
    user: str
    schema: type[BaseModel] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    tier: Tier = "quick"
    max_tokens: int | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
        )


@dataclass
class LLMResponse:
    text: str
    parsed: BaseModel | None = None
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_s: float = 0.0
    stop_reason: str | None = None

    def require(self, schema: type[T]) -> T:
        if not isinstance(self.parsed, schema):
            raise StructuredOutputError(
                f"expected {schema.__name__}, got {type(self.parsed).__name__}: {self.text[:200]}"
            )
        return self.parsed


class StructuredOutputError(RuntimeError):
    """The model did not return output matching the requested schema."""


class LLMRefusalError(RuntimeError):
    """The model declined the request (stop_reason == 'refusal')."""


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


# Public first-party list prices, USD per million tokens (input, output).
# Used only for cost estimates in run reports; unknown models cost 0.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
    "heuristic": (0.0, 0.0),
}


@dataclass
class CostLedger:
    """Accumulates usage by (role, model) so a run can report calls, tokens and cost."""

    calls: int = 0
    by_role: dict[str, Usage] = field(default_factory=lambda: defaultdict(Usage))
    by_model: dict[str, Usage] = field(default_factory=lambda: defaultdict(Usage))
    latency_s: float = 0.0

    def record(self, request: LLMRequest, response: LLMResponse) -> None:
        self.calls += 1
        self.by_role[request.role] = self.by_role[request.role] + response.usage
        self.by_model[response.model or "unknown"] = (
            self.by_model[response.model or "unknown"] + response.usage
        )
        self.latency_s += response.latency_s

    @property
    def total(self) -> Usage:
        acc = Usage()
        for u in self.by_model.values():
            acc = acc + u
        return acc

    def estimated_cost_usd(self) -> float:
        cost = 0.0
        for model, u in self.by_model.items():
            pin, pout = PRICE_TABLE.get(model, (0.0, 0.0))
            cost += u.input_tokens / 1e6 * pin + u.output_tokens / 1e6 * pout
        return cost

    def summary(self) -> dict[str, Any]:
        t = self.total
        return {
            "calls": self.calls,
            "input_tokens": t.input_tokens,
            "output_tokens": t.output_tokens,
            "cache_read_tokens": t.cache_read_tokens,
            "latency_s": round(self.latency_s, 3),
            "estimated_cost_usd": round(self.estimated_cost_usd(), 4),
        }


class TimedCall:
    """Context manager measuring wall-clock latency for a provider call."""

    def __enter__(self) -> TimedCall:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self.start
