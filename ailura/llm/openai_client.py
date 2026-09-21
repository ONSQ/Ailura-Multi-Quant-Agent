"""OpenAI-compatible chat client (OpenAI, Ollama, vLLM, OpenRouter, ...).

Structured output is requested via JSON mode plus the schema in the prompt and
validated locally, so it works on any endpoint that speaks the chat API.
"""

from __future__ import annotations

from typing import Any

from ailura.llm.base import LLMRequest, LLMResponse, StructuredOutputError, TimedCall, Usage
from ailura.llm.structured import parse_into, schema_instructions


class OpenAICompatibleLLM:
    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        max_tokens: int = 16000,
        temperature: float = 0.2,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=base_url) if base_url else OpenAI()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, request: LLMRequest) -> LLMResponse:
        system = request.system
        kwargs: dict[str, Any] = {}
        if request.schema is not None:
            system = f"{system}\n\n{schema_instructions(request.schema)}"
            kwargs["response_format"] = {"type": "json_object"}
        with TimedCall() as t:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=request.max_tokens or self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": request.user},
                ],
                **kwargs,
            )
        text = resp.choices[0].message.content or ""
        parsed = None
        if request.schema is not None:
            try:
                parsed = parse_into(request.schema, text)
            except StructuredOutputError:
                parsed = None
        u = getattr(resp, "usage", None)
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
        )
        return LLMResponse(
            text=text,
            parsed=parsed,
            usage=usage,
            model=getattr(resp, "model", self.model),
            latency_s=t.elapsed,
        )
