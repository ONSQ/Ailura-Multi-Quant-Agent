"""Market-data provider protocol and the point-in-time (look-ahead) guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Protocol, runtime_checkable

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class NewsItem:
    published: date
    headline: str
    summary: str = ""
    source: str = ""
    sentiment: float | None = None  # -1..1 if the source scores it


@dataclass(frozen=True)
class SocialPost:
    posted: date
    text: str
    platform: str = ""
    engagement: int = 0
    sentiment: float | None = None


@dataclass
class FundamentalsSnapshot:
    as_of: date
    metrics: dict[str, float] = field(default_factory=dict)  # pe, ps, eps growth, margins ...
    notes: list[str] = field(default_factory=list)


class LookaheadError(RuntimeError):
    """Raised when a caller asks for data after the decision timestamp."""


@runtime_checkable
class MarketDataProvider(Protocol):
    name: str

    def get_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Daily bars, index = DatetimeIndex (tz-naive), columns = OHLCV_COLUMNS, inclusive of end."""
        ...

    def get_news(self, ticker: str, as_of: date, lookback_days: int) -> list[NewsItem]: ...

    def get_social(self, ticker: str, as_of: date, lookback_days: int) -> list[SocialPost]: ...

    def get_fundamentals(self, ticker: str, as_of: date) -> FundamentalsSnapshot: ...


def to_date(d: date | datetime | str | pd.Timestamp) -> date:
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return pd.Timestamp(d).date()


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, sort, tz-naive DatetimeIndex, float dtypes."""
    out = df.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    if "adj_close" in out.columns and "close" not in out.columns:
        out["close"] = out["adj_close"]
    if not isinstance(out.index, pd.DatetimeIndex):
        if "date" in out.columns:
            out = out.set_index("date")
        out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing}")
    out = out[OHLCV_COLUMNS].astype(float)
    out.index.name = "date"
    return out


class PointInTimeProvider:
    """Wraps a provider so nothing after ``as_of`` can leak into a decision.

    Every call is clipped to ``as_of``; asking for data beyond it raises
    :class:`LookaheadError` (strict mode) or is silently clipped (lenient).
    Results are cached per (ticker, as_of) so repeated analyst calls are free.
    """

    def __init__(self, inner: MarketDataProvider, as_of: date, strict: bool = True) -> None:
        self.inner = inner
        self.as_of = to_date(as_of)
        self.strict = strict
        self.name = f"pit({inner.name})"
        self._cache: dict[tuple, object] = {}

    def _check(self, end: date, what: str) -> date:
        end = to_date(end)
        if end > self.as_of:
            if self.strict:
                raise LookaheadError(f"{what}: requested {end} but decision date is {self.as_of}")
            return self.as_of
        return end

    def get_ohlcv(self, ticker: str, start: date, end: date | None = None) -> pd.DataFrame:
        end = self._check(end or self.as_of, "ohlcv")
        key = ("ohlcv", ticker, to_date(start), end)
        if key not in self._cache:
            df = normalize_ohlcv(self.inner.get_ohlcv(ticker, to_date(start), end))
            df = df[df.index.date <= end]
            self._cache[key] = df
        return self._cache[key]  # type: ignore[return-value]

    def history(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        start = self.as_of - timedelta(days=int(lookback_days * 1.6) + 10)
        return self.get_ohlcv(ticker, start, self.as_of).tail(lookback_days)

    def get_news(self, ticker: str, as_of: date | None = None, lookback_days: int = 7) -> list[NewsItem]:
        as_of = self._check(as_of or self.as_of, "news")
        key = ("news", ticker, as_of, lookback_days)
        if key not in self._cache:
            items = [n for n in self.inner.get_news(ticker, as_of, lookback_days) if n.published <= as_of]
            self._cache[key] = items
        return self._cache[key]  # type: ignore[return-value]

    def get_social(self, ticker: str, as_of: date | None = None, lookback_days: int = 7) -> list[SocialPost]:
        as_of = self._check(as_of or self.as_of, "social")
        key = ("social", ticker, as_of, lookback_days)
        if key not in self._cache:
            items = [p for p in self.inner.get_social(ticker, as_of, lookback_days) if p.posted <= as_of]
            self._cache[key] = items
        return self._cache[key]  # type: ignore[return-value]

    def get_fundamentals(self, ticker: str, as_of: date | None = None) -> FundamentalsSnapshot:
        as_of = self._check(as_of or self.as_of, "fundamentals")
        key = ("fund", ticker, as_of)
        if key not in self._cache:
            snap = self.inner.get_fundamentals(ticker, as_of)
            if snap.as_of > as_of:
                raise LookaheadError(f"fundamentals dated {snap.as_of} after {as_of}")
            self._cache[key] = snap
        return self._cache[key]  # type: ignore[return-value]
