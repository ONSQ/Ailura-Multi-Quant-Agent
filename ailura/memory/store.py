"""Outcome-based decision memory (append-only JSONL) with point-in-time recall.

Each run records its final decision. Later, once the holding window has closed,
:meth:`DecisionMemory.resolve_pending` fetches the realized return and alpha
versus the benchmark and asks the :class:`~ailura.memory.reflection.Reflector`
for a lesson. Recall filters to lessons *resolved before the decision date*,
so a historical backtest can never learn from its own future.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ailura.data.base import MarketDataProvider, normalize_ohlcv, to_date


@dataclass
class MemoryEntry:
    id: str
    ticker: str
    decision_date: str
    rating: str
    conviction: float
    position_fraction: float
    executive_summary: str
    thesis: str
    holding_days: int
    tags: list[str] = field(default_factory=list)
    outcome: dict[str, Any] | None = None   # {"resolved_date", "raw_return", "benchmark_return", "alpha"}
    lesson: dict[str, Any] | None = None    # ReflectionLesson.model_dump()

    @property
    def resolved(self) -> bool:
        return self.outcome is not None and self.lesson is not None


class DecisionMemory:
    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None
        self.entries: list[MemoryEntry] = []
        if self.path and self.path.exists():
            with self.path.open() as f:
                for line in f:
                    if line.strip():
                        self.entries.append(MemoryEntry(**json.loads(line)))

    # ---------------------------------------------------------------- persist
    def _flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            for e in self.entries:
                f.write(json.dumps(asdict(e), default=str) + "\n")

    def record(self, *, ticker: str, decision_date: date, rating: str, conviction: float,
               position_fraction: float, executive_summary: str, thesis: str,
               holding_days: int, tags: list[str] | None = None) -> MemoryEntry:
        e = MemoryEntry(
            id=uuid.uuid4().hex[:12], ticker=ticker.upper(), decision_date=str(to_date(decision_date)),
            rating=rating, conviction=conviction, position_fraction=position_fraction,
            executive_summary=executive_summary, thesis=thesis, holding_days=holding_days,
            tags=sorted(set(tags or [])),
        )
        self.entries.append(e)
        self._flush()
        return e

    # ----------------------------------------------------------------- recall
    def recall(self, ticker: str, as_of: date, k: int = 3, tags: list[str] | None = None) -> list[dict[str, Any]]:
        """Lessons resolved strictly before ``as_of``, ranked by ticker match and tag overlap."""
        as_of = to_date(as_of)
        tagset = set(tags or [])
        scored: list[tuple[float, MemoryEntry]] = []
        for e in self.entries:
            if not e.resolved:
                continue
            if to_date(e.outcome["resolved_date"]) >= as_of:  # type: ignore[index]
                continue
            score = 2.0 if e.ticker == ticker.upper() else 0.0
            score += len(tagset & set(e.tags))
            # recency: newer lessons first as tie-break
            score += 1e-3 * (to_date(e.decision_date).toordinal() / 1e6)
            scored.append((score, e))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            {
                "ticker": e.ticker,
                "decision_date": e.decision_date,
                "rating": e.rating,
                "alpha": e.outcome["alpha"],  # type: ignore[index]
                "verdict": e.lesson["verdict"],  # type: ignore[index]
                "lesson": e.lesson["lesson"],  # type: ignore[index]
                "tags": e.tags,
            }
            for _, e in scored[:k]
        ]

    # ---------------------------------------------------------------- resolve
    def pending(self) -> list[MemoryEntry]:
        return [e for e in self.entries if not e.resolved]

    def resolve_pending(self, provider: MarketDataProvider, reflector, benchmark: str = "SPY",
                        as_of: date | None = None) -> int:
        """Score every pending decision whose holding window has closed (<= ``as_of``)."""
        n = 0
        for e in self.pending():
            d0 = to_date(e.decision_date)
            horizon_end = d0 + pd.tseries.offsets.BDay(e.holding_days + 3)
            horizon_end = horizon_end.date()
            if as_of is not None and horizon_end > to_date(as_of):
                continue
            try:
                px = normalize_ohlcv(provider.get_ohlcv(e.ticker, d0, horizon_end))
                bm = normalize_ohlcv(provider.get_ohlcv(benchmark, d0, horizon_end))
            except Exception:
                continue
            px = px[px.index.date >= d0]
            bm = bm[bm.index.date >= d0]
            if len(px) < e.holding_days + 1 or len(bm) < e.holding_days + 1:
                continue
            raw = float(px["close"].iloc[e.holding_days] / px["close"].iloc[0] - 1)
            bmr = float(bm["close"].iloc[e.holding_days] / bm["close"].iloc[0] - 1)
            resolved_date = px.index[e.holding_days].date()
            e.outcome = {
                "resolved_date": str(resolved_date), "raw_return": round(raw, 6),
                "benchmark_return": round(bmr, 6), "alpha": round(raw - bmr, 6),
            }
            lesson = reflector.reflect(entry=e)
            e.lesson = lesson.model_dump()
            n += 1
        if n:
            self._flush()
        return n

    def stats(self) -> dict[str, Any]:
        res = [e for e in self.entries if e.resolved]
        by_rating: dict[str, dict[str, float]] = {}
        for e in res:
            b = by_rating.setdefault(e.rating, {"n": 0, "hits": 0, "alpha_sum": 0.0})
            a = e.outcome["alpha"]  # type: ignore[index]
            b["n"] += 1
            b["alpha_sum"] += a
            if e.rating in ("Buy", "Overweight") and a > 0 or e.rating in ("Sell", "Underweight") and a < 0:
                b["hits"] += 1
        return {
            "total": len(self.entries), "resolved": len(res), "pending": len(self.entries) - len(res),
            "by_rating": {r: {"n": b["n"], "hit_rate": (b["hits"] / b["n"]) if b["n"] and r != "Hold" else None,
                              "mean_alpha": b["alpha_sum"] / b["n"] if b["n"] else None} for r, b in by_rating.items()},
        }
