"""Typed pipeline state shared by every agent."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ailura.agents.schemas import (
    AnalystReport,
    DebateArgument,
    FinalDecision,
    ResearchVerdict,
    RiskArgument,
    TraderProposal,
)
from ailura.llm.base import CostLedger


class PortfolioContext(BaseModel):
    """What the trader/risk team knows about the book for this ticker."""

    current_weight: float = 0.0  # signed fraction of max allocation currently held
    days_held: int = 0  # trading days since the current position was opened
    cash_fraction: float = 1.0
    max_position_fraction: float = 1.0
    allow_short: bool = False
    target_annual_vol: float = 0.20
    unrealized_pnl_pct: float | None = None
    notes: list[str] = Field(default_factory=list)


class TradingState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ticker: str
    trade_date: date
    # raw inputs (point-in-time)
    features: dict[str, Any] = Field(default_factory=dict)
    recent_bars: list[dict[str, Any]] = Field(default_factory=list)
    news: list[dict[str, Any]] = Field(default_factory=list)
    social: list[dict[str, Any]] = Field(default_factory=list)
    fundamentals: dict[str, Any] = Field(default_factory=dict)
    memories: list[dict[str, Any]] = Field(default_factory=list)
    portfolio: PortfolioContext = Field(default_factory=PortfolioContext)
    # agent outputs
    reports: dict[str, AnalystReport] = Field(default_factory=dict)
    debate: list[DebateArgument] = Field(default_factory=list)
    debate_rounds_run: int = 0
    debate_stopped_early: bool = False
    research_verdict: ResearchVerdict | None = None
    trader_proposal: TraderProposal | None = None
    risk_debate: list[RiskArgument] = Field(default_factory=list)
    risk_rounds_run: int = 0
    final_decision: FinalDecision | None = None
    # bookkeeping
    transcript: list[dict[str, Any]] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    ledger: CostLedger = Field(default_factory=CostLedger, exclude=True)

    # -- helpers used by prompts --------------------------------------------
    def report_digest(self) -> list[dict[str, Any]]:
        return [
            {
                "analyst": r.analyst,
                "stance": r.stance,
                "score": r.score,
                "confidence": r.confidence,
                "data_quality": r.data_quality,
                "summary": r.summary,
                "signals": [s.model_dump() for s in r.signals[:6]],
            }
            for r in self.reports.values()
        ]

    def debate_digest(self) -> list[dict[str, Any]]:
        return [a.model_dump() for a in self.debate]

    def risk_digest(self) -> list[dict[str, Any]]:
        return [a.model_dump() for a in self.risk_debate]

    @property
    def close(self) -> float | None:
        return self.features.get("close")

    def to_record(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["cost"] = self.ledger.summary()
        return d
