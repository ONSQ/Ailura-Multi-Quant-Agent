"""Local CSV provider: ``<data_dir>/<TICKER>.csv`` with Date,Open,High,Low,Close,Volume.

Optional feeds: ``<TICKER>_news.jsonl`` ({"date","headline","summary","source","sentiment"}),
``<TICKER>_social.jsonl`` ({"date","text","platform","engagement","sentiment"}),
``<TICKER>_fundamentals.json`` ([{"as_of": "YYYY-MM-DD", "metrics": {...}}, ...]).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ailura.data.base import FundamentalsSnapshot, NewsItem, SocialPost, normalize_ohlcv, to_date


class CSVProvider:
    name = "csv"

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self._cache: dict[str, pd.DataFrame] = {}

    def _load(self, ticker: str) -> pd.DataFrame:
        t = ticker.upper()
        if t not in self._cache:
            path = self.data_dir / f"{t}.csv"
            if not path.exists():
                raise FileNotFoundError(f"no price file for {t}: {path}")
            self._cache[t] = normalize_ohlcv(pd.read_csv(path))
        return self._cache[t]

    def get_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        return self._load(ticker).loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()

    def _jsonl(self, name: str) -> list[dict]:
        path = self.data_dir / name
        if not path.exists():
            return []
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def get_news(self, ticker: str, as_of: date, lookback_days: int) -> list[NewsItem]:
        lo = as_of - timedelta(days=lookback_days)
        out = []
        for row in self._jsonl(f"{ticker.upper()}_news.jsonl"):
            d = to_date(row["date"])
            if lo <= d <= as_of:
                out.append(NewsItem(d, row.get("headline", ""), row.get("summary", ""), row.get("source", ""), row.get("sentiment")))
        return sorted(out, key=lambda n: n.published, reverse=True)

    def get_social(self, ticker: str, as_of: date, lookback_days: int) -> list[SocialPost]:
        lo = as_of - timedelta(days=lookback_days)
        out = []
        for row in self._jsonl(f"{ticker.upper()}_social.jsonl"):
            d = to_date(row["date"])
            if lo <= d <= as_of:
                out.append(SocialPost(d, row.get("text", ""), row.get("platform", ""), int(row.get("engagement", 0)), row.get("sentiment")))
        return sorted(out, key=lambda p: p.posted, reverse=True)

    def get_fundamentals(self, ticker: str, as_of: date) -> FundamentalsSnapshot:
        path = self.data_dir / f"{ticker.upper()}_fundamentals.json"
        if not path.exists():
            return FundamentalsSnapshot(as_of=as_of, metrics={}, notes=["no fundamentals file"])
        snaps = json.loads(path.read_text())
        eligible = [s for s in snaps if to_date(s["as_of"]) <= as_of]
        if not eligible:
            return FundamentalsSnapshot(as_of=as_of, metrics={}, notes=["no fundamentals as of date"])
        latest = max(eligible, key=lambda s: to_date(s["as_of"]))
        return FundamentalsSnapshot(as_of=to_date(latest["as_of"]), metrics={k: float(v) for k, v in latest["metrics"].items()}, notes=latest.get("notes", []))
