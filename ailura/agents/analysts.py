"""Analyst team: market (technical), news, social sentiment, fundamentals.

Unlike the upstream implementation, analysts do not run an agentic tool loop:
the orchestrator pre-fetches every point-in-time input once and hands it over.
That removes 4-8 LLM round-trips per run and makes the data path auditable.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ailura.agents.base import Agent
from ailura.agents.schemas import AnalystReport
from ailura.graph.state import TradingState


class MarketAnalyst(Agent[AnalystReport]):
    role: ClassVar[str] = "market_analyst"
    schema = AnalystReport
    system_prompt = (
        "You are the Market (technical) Analyst. You receive a computed indicator snapshot "
        "(moving averages, MACD, RSI, Bollinger, ATR, VWMA, KDJ, z-score, realized vol, "
        "60-day support/resistance) and the last bars. Select the most informative, "
        "non-redundant indicators, describe trend, momentum, volatility regime and key "
        "levels, and give a stance with a signed score. Take direction from mom_12_1 / mom_6_1 "
        "(momentum that skips the latest month) and the trend, NOT from the last few days: RSI, "
        "MACD histogram and ret_5d describe what already happened. Use them for entry_timing "
        "only; a sharp move in the trend's direction means wait for a pullback, not chase. "
        "Set analyst='market'."
    )

    def build_context(self, state: TradingState) -> dict[str, Any]:
        return {
            "ticker": state.ticker,
            "date": str(state.trade_date),
            "features": state.features,
            "recent_bars": state.recent_bars[-10:],
        }


class NewsAnalyst(Agent[AnalystReport]):
    role: ClassVar[str] = "news_analyst"
    schema = AnalystReport
    system_prompt = (
        "You are the News Analyst. Read the recent company and macro headlines, weigh their "
        "materiality and recency, and summarise the likely impact on the stock over the next "
        "1-4 weeks. Distinguish confirmed facts from speculation. Most headlines report a move "
        "that already happened: compare the news with the price reaction already seen "
        "(ret_5d, stretch_5d = last week's move in units of its normal size). Good news after a "
        "big rally is priced in, not bullish; score only the part the price has NOT yet "
        "reflected. Set analyst='news'."
    )

    def build_context(self, state: TradingState) -> dict[str, Any]:
        sents = [n["sentiment"] for n in state.news if n.get("sentiment") is not None]
        return {
            "ticker": state.ticker,
            "date": str(state.trade_date),
            "news_count": len(state.news),
            "avg_sentiment": round(float(np.mean(sents)), 4) if sents else None,
            "ret_5d": state.features.get("ret_5d"),
            "stretch_5d": state.features.get("stretch_5d"),
            "news": state.news[:25],
        }


class SentimentAnalyst(Agent[AnalystReport]):
    role: ClassVar[str] = "sentiment_analyst"
    schema = AnalystReport
    system_prompt = (
        "You are the Social Sentiment Analyst. Assess the mood of retail and social media "
        "discussion, engagement-weighted. Crowd mood mostly echoes the last move (see ret_5d and "
        "stretch_5d): score only mood the price action does not already explain, in either "
        "direction. Neither chase nor reflexively fade it. Set analyst='sentiment'."
    )

    def build_context(self, state: TradingState) -> dict[str, Any]:
        posts = state.social
        s = [p["sentiment"] for p in posts if p.get("sentiment") is not None]
        w = [max(1, p.get("engagement", 1)) for p in posts if p.get("sentiment") is not None]
        weighted = float(np.average(s, weights=w)) if s else None
        return {
            "ticker": state.ticker,
            "date": str(state.trade_date),
            "post_count": len(posts),
            "avg_sentiment": round(float(np.mean(s)), 4) if s else None,
            "engagement_weighted_sentiment": round(weighted, 4) if weighted is not None else None,
            "sample_posts": posts[:20],
            "ret_5d": state.features.get("ret_5d"),
            "stretch_5d": state.features.get("stretch_5d"),
        }


class FundamentalsAnalyst(Agent[AnalystReport]):
    role: ClassVar[str] = "fundamentals_analyst"
    schema = AnalystReport
    system_prompt = (
        "You are the Fundamentals Analyst. Evaluate valuation, growth, profitability and "
        "balance-sheet metrics from the latest available filing (note the as-of date and any "
        "staleness). Judge whether fundamentals support or contradict the current price. "
        "Set analyst='fundamentals'."
    )

    def build_context(self, state: TradingState) -> dict[str, Any]:
        return {
            "ticker": state.ticker,
            "date": str(state.trade_date),
            "fundamentals": state.fundamentals,
            "close": state.features.get("close"),
        }


ANALYSTS: dict[str, type[Agent[AnalystReport]]] = {
    "market": MarketAnalyst,
    "news": NewsAnalyst,
    "sentiment": SentimentAnalyst,
    "fundamentals": FundamentalsAnalyst,
}
