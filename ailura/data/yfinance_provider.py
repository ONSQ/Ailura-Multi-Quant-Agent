"""Yahoo Finance provider (optional dependency ``yfinance``).

Prices are cached to ``<data_dir>/cache/<TICKER>.csv`` so a backtest hits the
network once. News/social/fundamentals from Yahoo are *not* point-in-time
(they reflect the present): for a decision date older than
``LIVE_WINDOW_DAYS`` the provider returns empty fundamentals instead of
today's numbers stamped with a past date (which silently leaked the future
into backtests). Use :class:`CSVProvider` feeds for honest historical text data.

For live use, a cache that is behind the last completed session is refreshed,
and today's bar is dropped until the US close (a partial intraday bar is not a
close).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ailura.data.base import FundamentalsSnapshot, NewsItem, SocialPost, normalize_ohlcv

LIVE_WINDOW_DAYS = 3
_NY = ZoneInfo("America/New_York")
_CLOSE_SETTLED = time(16, 30)


def last_completed_session(now: datetime | None = None) -> date:
    """Most recent weekday whose US session has closed (holidays not modelled)."""
    now = now or datetime.now(_NY)
    d = now.date() if now.time() >= _CLOSE_SETTLED else now.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


class YFinanceProvider:
    name = "yfinance"

    def __init__(self, data_dir: Path | str = "./data") -> None:
        self.cache_dir = Path(data_dir) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, pd.DataFrame] = {}
        self._fetched: set[tuple[str, date]] = set()

    def _load(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        t = ticker.upper()
        path = self.cache_dir / f"{t}.csv"
        if t in self._mem:
            df = self._mem[t]
        elif path.exists():
            df = normalize_ohlcv(pd.read_csv(path, index_col=0, parse_dates=True))
        else:
            df = pd.DataFrame()
        want_end = min(end, last_completed_session())
        need_fetch = df.empty or df.index.min().date() > start or df.index.max().date() < want_end
        if need_fetch and (t, want_end) not in self._fetched:
            self._fetched.add((t, want_end))  # one attempt per process (holidays leave a gap)
            import yfinance as yf

            lo = min(start, df.index.min().date()) if not df.empty else start
            raw = yf.download(t, start=lo - timedelta(days=5), end=end + timedelta(days=1), auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = normalize_ohlcv(raw)
            # never treat an in-progress bar as a close
            df = df[df.index.date <= last_completed_session()]
            df.to_csv(path)
        self._mem[t] = df
        return df

    def get_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        df = self._load(ticker, start, end)
        return df.loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()

    def get_news(self, ticker: str, as_of: date, lookback_days: int) -> list[NewsItem]:
        import yfinance as yf

        out: list[NewsItem] = []
        try:
            for item in yf.Ticker(ticker).news or []:
                content = item.get("content", item)
                pub = content.get("pubDate") or content.get("providerPublishTime")
                d = pd.Timestamp(pub, unit="s" if isinstance(pub, (int, float)) else None).date()
                if as_of - timedelta(days=lookback_days) <= d <= as_of:
                    out.append(NewsItem(d, content.get("title", ""), content.get("summary", ""), "yahoo"))
        except Exception:  # network / schema drift: degrade to empty feed
            pass
        return out

    def get_social(self, ticker: str, as_of: date, lookback_days: int) -> list[SocialPost]:
        return []  # Yahoo has no social feed; plug a CSV feed for historical posts

    def get_fundamentals(self, ticker: str, as_of: date) -> FundamentalsSnapshot:
        import yfinance as yf

        metrics: dict[str, float] = {}
        if (date.today() - as_of).days > LIVE_WINDOW_DAYS:
            return FundamentalsSnapshot(as_of=as_of, metrics={}, notes=["yahoo fundamentals are present-day only; withheld for a historical date to avoid look-ahead"])
        try:
            info = yf.Ticker(ticker).info or {}
            for src, dst in {
                "trailingPE": "pe_ratio",
                "priceToSalesTrailing12Months": "ps_ratio",
                "earningsGrowth": "eps_growth_yoy",
                "revenueGrowth": "revenue_growth_yoy",
                "grossMargins": "gross_margin",
            }.items():
                if info.get(src) is not None:
                    metrics[dst] = float(info[src])
        except Exception:
            pass
        return FundamentalsSnapshot(as_of=as_of, metrics=metrics, notes=["yahoo snapshot is NOT point-in-time"])
