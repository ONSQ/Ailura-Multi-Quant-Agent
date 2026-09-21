"""Structured outputs for every agent role.

The paper's agents exchange free-form reports; here every hand-off is a
validated pydantic object so downstream agents (and the backtester) never
depend on regex-scraping prose. ``conviction`` is a signed score in [-1, 1]
that drives position sizing, not just the direction.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Stance = Literal["bullish", "bearish", "neutral"]
Rating = Literal["Buy", "Overweight", "Hold", "Underweight", "Sell"]
Action = Literal["BUY", "SELL", "HOLD"]


class Signal(BaseModel):
    name: str = Field(description="Indicator or fact name, e.g. 'RSI(14)' or 'EPS growth YoY'")
    value: str = Field(description="Observed value as text, e.g. '62.3' or '+18%'")
    interpretation: str = Field(description="One-sentence read of what it implies")
    direction: Stance = "neutral"


class AnalystReport(BaseModel):
    analyst: str
    summary: str = Field(description="3-6 sentence narrative report")
    signals: list[Signal] = Field(default_factory=list, max_length=12)
    stance: Stance = "neutral"
    score: float = Field(0.0, ge=-1, le=1, description="Signed strength of the view")
    confidence: float = Field(0.5, ge=0, le=1)
    data_quality: float = Field(1.0, ge=0, le=1, description="1 = complete data, 0 = nothing usable")


class DebateArgument(BaseModel):
    side: Literal["bull", "bear"]
    thesis: str
    points: list[str] = Field(default_factory=list, max_length=8)
    rebuttals: list[str] = Field(default_factory=list, max_length=8, description="Direct responses to the other side's latest points")
    conceded: list[str] = Field(default_factory=list, max_length=4, description="Opponent points you accept")
    confidence: float = Field(0.5, ge=0, le=1)


class ResearchVerdict(BaseModel):
    rating: Rating
    conviction: float = Field(ge=-1, le=1)
    rationale: str
    decisive_arguments: list[str] = Field(default_factory=list, max_length=6)
    key_risks: list[str] = Field(default_factory=list, max_length=6)
    plan: str = Field(description="Strategic actions: sizing relative to standard, triggers to re-evaluate")


class TraderProposal(BaseModel):
    action: Action
    rating: Rating
    conviction: float = Field(ge=-1, le=1)
    entry_price: float | None = Field(None, description="Absolute price in quote currency")
    stop_loss: float | None = None
    take_profit: float | None = None
    position_fraction: float = Field(0.0, ge=-1, le=1, description="Fraction of max allocation, signed")
    horizon_days: int = Field(5, ge=1, le=250)
    reasoning: str


class RiskArgument(BaseModel):
    persona: Literal["aggressive", "conservative", "neutral"]
    stance_on_proposal: Literal["support", "scale_down", "scale_up", "reject"]
    points: list[str] = Field(default_factory=list, max_length=8)
    rebuttals: list[str] = Field(default_factory=list, max_length=8)
    recommended_position_fraction: float = Field(ge=-1, le=1)
    recommended_stop_loss: float | None = None
    confidence: float = Field(0.5, ge=0, le=1)


class FinalDecision(BaseModel):
    rating: Rating
    action: Action
    conviction: float = Field(ge=-1, le=1)
    position_fraction: float = Field(ge=-1, le=1)
    stop_loss: float | None = None
    take_profit: float | None = None
    executive_summary: str
    thesis: str
    invalidation_triggers: list[str] = Field(default_factory=list, max_length=6)
    lessons_applied: list[str] = Field(default_factory=list, max_length=4)


class ReflectionLesson(BaseModel):
    verdict: Literal["correct", "incorrect", "inconclusive"]
    what_the_outcome_shows: str
    thesis_component_affected: str
    lesson: str = Field(description="One concrete, reusable lesson (<= 40 words)")
    tags: list[str] = Field(default_factory=list, max_length=6)
