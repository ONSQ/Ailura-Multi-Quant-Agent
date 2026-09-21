"""Anthropic Claude client (official SDK).

Uses ``client.messages.parse`` with a pydantic ``output_format`` for structured
outputs, adaptive thinking, and an ``effort`` level. Server-side refusal
fallbacks are enabled by default (``fallbacks="default"`` under the
``server-side-fallback-2026-07-01`` beta); disable with
``AiluraConfig.anthropic_fallbacks=False``.
"""

from __future__ import annotations

from typing import Any

from ailura.llm.base import (
    LLMRefusalError,
    LLMRequest,
    LLMResponse,
    StructuredOutputError,
    TimedCall,
    Usage,
)
from ailura.llm.structured import parse_into

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5",
        effort: str = "high",
        max_tokens: int = 16000,
        fallbacks: bool = True,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        if client is None:
            import anthropic  # local import keeps the SDK optional

            client = anthropic.Anthropic(max_retries=max_retries)
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks

    def _common_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens or self.max_tokens,
            "system": [
                {"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": request.user}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
        }
        if self.fallbacks:
            kwargs["extra_headers"] = {"anthropic-beta": FALLBACK_BETA}
            kwargs["extra_body"] = {"fallbacks": "default"}
        return kwargs

    def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs = self._common_kwargs(request)
        with TimedCall() as t:
            if request.schema is not None:
                resp = self.client.messages.parse(output_format=request.schema, **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        if getattr(resp, "stop_reason", None) == "refusal":
            details = getattr(resp, "stop_details", None)
            raise LLMRefusalError(f"model refused request for role={request.role}: {details}")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        parsed = None
        if request.schema is not None:
            parsed = getattr(resp, "parsed_output", None)
            if parsed is None:
                try:
                    parsed = parse_into(request.schema, text)
                except StructuredOutputError:
                    parsed = None
        u = getattr(resp, "usage", None)
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        )
        return LLMResponse(
            text=text,
            parsed=parsed,
            usage=usage,
            model=getattr(resp, "model", self.model),
            latency_s=t.elapsed,
            stop_reason=getattr(resp, "stop_reason", None),
        )
