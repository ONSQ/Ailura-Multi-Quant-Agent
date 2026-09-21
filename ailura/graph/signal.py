"""Deterministic 5-tier rating parsing and mapping to actions / target weights.

A parse failure yields ``REVIEW`` so it can never masquerade as a tradeable
neutral signal (the upstream project's design; kept here on purpose).
"""

from __future__ import annotations

import re
import unicodedata

RATINGS: tuple[str, ...] = ("Buy", "Overweight", "Hold", "Underweight", "Sell")
RATING_REVIEW = "REVIEW"
_RATING_SET = {r.lower(): r for r in RATINGS}

_LABELED = re.compile(
    r"(?im)^\W*(?:final\s+)?(?:rating|recommendation|decision|action)\b[^:\-–—]*[:\-–—]\s*\**\s*"
    r"(buy|overweight|hold|underweight|sell)\b"
)
_WORD = re.compile(r"(?i)\b(buy|overweight|hold|underweight|sell)\b")


def extract_rating(text: str) -> str | None:
    """Return a canonical rating from free text, or None if ambiguous/missing."""
    if not text:
        return None
    norm = unicodedata.normalize("NFKC", text)
    labeled = [m.group(1).lower() for m in _LABELED.finditer(norm)]
    if labeled:
        return _RATING_SET[labeled[-1]]
    words = {m.group(1).lower() for m in _WORD.finditer(norm)}
    if len(words) == 1:
        return _RATING_SET[words.pop()]
    return None


def parse_rating(text: str, default: str = RATING_REVIEW) -> str:
    return extract_rating(text) or default


def is_review(signal: str) -> bool:
    return signal == RATING_REVIEW


def rating_to_action(rating: str) -> str:
    """Collapse the 5-tier rating to BUY / SELL / HOLD (paper's action space)."""
    if rating in ("Buy", "Overweight"):
        return "BUY"
    if rating in ("Sell", "Underweight"):
        return "SELL"
    return "HOLD"


def rating_to_weight(rating: str, allow_short: bool = False) -> float:
    """Default target weight per rating (long-only unless ``allow_short``)."""
    table = {"Buy": 1.0, "Overweight": 0.6, "Hold": 0.0, "Underweight": -0.4, "Sell": -1.0}
    w = table.get(rating, 0.0)
    if not allow_short and w < 0:
        return 0.0
    return w


def conviction_to_rating(conviction: float) -> str:
    """Map a conviction score in [-1, 1] to the 5-tier scale."""
    if conviction >= 0.6:
        return "Buy"
    if conviction >= 0.2:
        return "Overweight"
    if conviction > -0.2:
        return "Hold"
    if conviction > -0.6:
        return "Underweight"
    return "Sell"
