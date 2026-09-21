"""Configuration for Ailura.

Every field can be overridden with an ``AILURA_<FIELD_UPPER>`` environment
variable (booleans accept true/false/1/0/yes/no). Invalid values fail loudly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

LLMProvider = Literal["heuristic", "anthropic", "openai"]
DataProviderName = Literal["synthetic", "csv", "yfinance"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class AiluraConfig(BaseModel):
    # --- LLM -------------------------------------------------------------
    llm_provider: LLMProvider = "heuristic"
    deep_model: str = "claude-opus-5"
    quick_model: str = "claude-sonnet-5"
    deep_effort: Effort = "high"
    quick_effort: Effort = "low"
    max_output_tokens: int = 16000
    anthropic_fallbacks: bool = True
    openai_base_url: str | None = None
    llm_max_retries: int = 2

    # --- Data ------------------------------------------------------------
    data_provider: DataProviderName = "synthetic"
    data_dir: Path = Path("./data")
    lookback_days: int = 260
    news_lookback_days: int = 7
    benchmark_ticker: str = "SPY"
    synthetic_seed: int = 7

    # --- Workflow ------------------------------------------------------------
    analysts: list[str] = Field(default_factory=lambda: ["market", "news", "sentiment", "fundamentals"])
    max_debate_rounds: int = 2
    max_risk_rounds: int = 1
    early_stop_debate: bool = True
    debate_novelty_threshold: float = 0.35
    memory_k: int = 3
    holding_days: int = 5

    # --- Portfolio / risk --------------------------------------------------
    target_annual_vol: float = 0.20
    max_position_fraction: float = 1.0
    allow_short: bool = False
    commission_bps: float = 1.0
    slippage_bps: float = 2.0

    # --- Paths -------------------------------------------------------------
    results_dir: Path = Path("./results")
    memory_path: Path | None = Path("./.ailura/memory.jsonl")  # None = in-memory only

    @field_validator("max_debate_rounds", "max_risk_rounds", "memory_k", "holding_days", "lookback_days")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @classmethod
    def from_env(cls, prefix: str = "AILURA_", **overrides) -> AiluraConfig:
        """Build a config from environment variables plus explicit overrides."""
        values: dict = {}
        for name, field in cls.model_fields.items():
            env_key = f"{prefix}{name.upper()}"
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            default = field.default if field.default_factory is None else field.default_factory()
            values[name] = _coerce(raw, default, name)
        values.update(overrides)
        return cls(**values)


def _coerce(raw: str, default, name: str):
    if isinstance(default, bool):
        low = raw.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name}: expected boolean, got {raw!r}")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, Path):
        return Path(raw)
    if isinstance(default, list):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return raw
