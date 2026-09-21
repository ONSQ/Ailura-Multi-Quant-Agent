from __future__ import annotations

from ailura.config import AiluraConfig
from ailura.llm.base import LLMClient


def build_llm(config: AiluraConfig, tier: str = "quick") -> LLMClient:
    """Build the client for a tier ("quick" or "deep") from config."""
    model = config.deep_model if tier == "deep" else config.quick_model
    effort = config.deep_effort if tier == "deep" else config.quick_effort
    if config.llm_provider == "heuristic":
        from ailura.llm.heuristic import HeuristicLLM

        return HeuristicLLM()
    if config.llm_provider == "anthropic":
        from ailura.llm.anthropic_client import AnthropicLLM

        return AnthropicLLM(
            model=model,
            effort=effort,
            max_tokens=config.max_output_tokens,
            fallbacks=config.anthropic_fallbacks,
            max_retries=config.llm_max_retries,
        )
    if config.llm_provider == "openai":
        from ailura.llm.openai_client import OpenAICompatibleLLM

        return OpenAICompatibleLLM(
            model=model, base_url=config.openai_base_url, max_tokens=config.max_output_tokens
        )
    raise ValueError(f"unknown llm_provider {config.llm_provider!r}")
