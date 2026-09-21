"""Deterministic synthetic market: regime-switching GBM plus derived news/social/fundamentals.

Prices for a ticker are a pure function of (seed, ticker), so tests and
offline backtests are reproducible. Text feeds are generated from the price
path (earnings beats after strong weeks, downgrades after drawdowns, ...) so
the analysts have *something correlated with reality* to reason about.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ailura.data.base import FundamentalsSnapshot, NewsItem, SocialPost, normalize_ohlcv

_BASE = date(2015, 1, 1)
_YEARS = 14


def _ticker_seed(seed: int, ticker: str) -> int:
    h = hashlib.sha256(f"{seed}:{ticker.upper()}".encode()).digest()
    return int.from_bytes(h[:4], "big")


class SyntheticProvider:
    name = "synthetic"

    def __init__(self, seed: int = 7, start_price: float = 100.0, annual_vol: float = 0.28) -> None:
        self.seed = seed
        self.start_price = start_price
        self.annual_vol = annual_vol
        self._paths: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ prices
    def _path(self, ticker: str) -> pd.DataFrame:
        t = ticker.upper()
        if t in self._paths:
            return self._paths[t]
        rng = np.random.default_rng(_ticker_seed(self.seed, t))
        idx = pd.bdate_range(_BASE, _BASE + timedelta(days=365 * _YEARS))
        n = len(idx)
        # two-state regime: bull (drift +, vol low) / bear (drift -, vol high)
        regime = np.zeros(n, dtype=int)
        p_switch = 1 / 90
        for i in range(1, n):
            regime[i] = regime[i - 1] if rng.random() > p_switch else 1 - regime[i - 1]
        daily_vol = self.annual_vol / np.sqrt(252) * np.where(regime == 0, 0.8, 1.5)
        drift = np.where(regime == 0, 0.12, -0.10) / 252
        shocks = rng.standard_normal(n) * daily_vol
        # add a little momentum (AR(1) in returns) so trend indicators mean something
        rets = np.empty(n)
        rets[0] = drift[0] + shocks[0]
        for i in range(1, n):
            rets[i] = drift[i] + 0.08 * rets[i - 1] + shocks[i]
        close = self.start_price * np.exp(np.cumsum(rets))
        intraday = np.abs(rng.standard_normal(n)) * daily_vol * close
        open_ = close * (1 + rng.standard_normal(n) * daily_vol * 0.5)
        high = np.maximum(open_, close) + intraday
        low = np.minimum(open_, close) - intraday
        volume = (rng.lognormal(15, 0.35, n) * (1 + 2 * np.abs(rets) / daily_vol)).round()
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
        )
        df.attrs["regime"] = regime
        self._paths[t] = normalize_ohlcv(df)
        self._paths[t].attrs["regime"] = regime
        return self._paths[t]

    def get_ohlcv(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        df = self._path(ticker)
        return df.loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()

    # ------------------------------------------------------------------- text
    def _recent_return(self, ticker: str, as_of: date, days: int) -> float:
        df = self._path(ticker).loc[: pd.Timestamp(as_of)]
        if len(df) < days + 1:
            return 0.0
        return float(df["close"].iloc[-1] / df["close"].iloc[-days - 1] - 1)

    def get_news(self, ticker: str, as_of: date, lookback_days: int) -> list[NewsItem]:
        t = ticker.upper()
        rng = np.random.default_rng(_ticker_seed(self.seed, f"{t}:news:{as_of}"))
        r5 = self._recent_return(t, as_of, 5)
        r20 = self._recent_return(t, as_of, 20)
        items: list[NewsItem] = []
        for k in range(lookback_days):
            d = as_of - timedelta(days=k)
            if d.weekday() >= 5:
                continue
            base = 0.6 * r20 / 0.1 + 0.4 * r5 / 0.05
            s = float(np.clip(base + rng.normal(0, 0.5), -1, 1))
            if s > 0.35:
                head, body = (
                    f"{t} shares extend rally as analysts lift price targets",
                    "Sell-side notes cite improving demand and margin expansion.",
                )
            elif s < -0.35:
                head, body = (
                    f"{t} slides after cautious guidance; downgrade from a major broker",
                    "Concerns over decelerating growth and competitive pressure weigh on sentiment.",
                )
            else:
                head, body = (
                    f"{t} little changed as investors await macro data",
                    "Traders position ahead of the next inflation print and central bank meeting.",
                )
            items.append(NewsItem(published=d, headline=head, summary=body, source="synthetic-wire", sentiment=s))
        return items

    def get_social(self, ticker: str, as_of: date, lookback_days: int) -> list[SocialPost]:
        t = ticker.upper()
        rng = np.random.default_rng(_ticker_seed(self.seed, f"{t}:social:{as_of}"))
        r5 = self._recent_return(t, as_of, 5)
        posts: list[SocialPost] = []
        for k in range(lookback_days * 3):
            d = as_of - timedelta(days=k // 3)
            s = float(np.clip(r5 / 0.04 + rng.normal(0, 0.8), -1, 1))
            text = (
                f"${t} looking strong, adding on dips" if s > 0.3
                else f"${t} chart broken, cutting exposure" if s < -0.3
                else f"${t} range-bound, waiting for a catalyst"
            )
            posts.append(SocialPost(posted=d, text=text, platform="synthetic-forum", engagement=int(rng.integers(5, 500)), sentiment=s))
        return posts

    def get_fundamentals(self, ticker: str, as_of: date) -> FundamentalsSnapshot:
        t = ticker.upper()
        rng = np.random.default_rng(_ticker_seed(self.seed, f"{t}:fund:{as_of.year}:{(as_of.month - 1) // 3}"))
        r60 = self._recent_return(t, as_of, 60)
        # Fundamentals are quarterly (as_of the last quarter end) to mimic reporting lag.
        q_end = date(as_of.year, 3 * ((as_of.month - 1) // 3) + 1, 1) - timedelta(days=1)
        metrics = {
            "pe_ratio": float(np.clip(24 + 40 * r60 + rng.normal(0, 3), 8, 80)),
            "ps_ratio": float(np.clip(5 + 8 * r60 + rng.normal(0, 0.7), 0.8, 20)),
            "eps_growth_yoy": float(np.clip(0.10 + 0.8 * r60 + rng.normal(0, 0.05), -0.5, 0.8)),
            "revenue_growth_yoy": float(np.clip(0.08 + 0.5 * r60 + rng.normal(0, 0.04), -0.3, 0.6)),
            "gross_margin": float(np.clip(0.45 + rng.normal(0, 0.03), 0.1, 0.9)),
            "net_debt_to_ebitda": float(np.clip(1.0 + rng.normal(0, 0.5), -2, 6)),
            "free_cash_flow_yield": float(np.clip(0.04 - 0.05 * r60 + rng.normal(0, 0.01), -0.02, 0.12)),
        }
        return FundamentalsSnapshot(as_of=q_end, metrics=metrics, notes=["synthetic quarterly snapshot"])
