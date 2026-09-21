"""Performance metrics (the paper reports CR%, AR%, Sharpe, MDD%; we add Sortino, Calmar, hit rate)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cumulative_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def annualized_return(equity: pd.Series, periods: int = TRADING_DAYS) -> float:
    n = len(equity) - 1
    if n <= 0 or equity.iloc[0] <= 0:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    if total <= 0:
        return -1.0
    return float(total ** (periods / n) - 1)


def sharpe(returns: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna() - rf_annual / periods
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def sortino(returns: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna() - rf_annual / periods
    downside = r[r < 0]
    if len(r) < 2 or len(downside) == 0 or downside.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / downside.std(ddof=1) * np.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(-dd.min())


def calmar(equity: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(equity)
    return float(annualized_return(equity, periods) / mdd) if mdd > 0 else 0.0


def hit_rate(returns: pd.Series) -> float:
    active = returns[returns != 0].dropna()
    return float((active > 0).mean()) if len(active) else 0.0


def summarize(equity: pd.Series, returns: pd.Series | None = None, trades: int | None = None) -> dict[str, float]:
    if returns is None:
        returns = equity.pct_change().dropna()
    out = {
        "cumulative_return": cumulative_return(equity),
        "annualized_return": annualized_return(equity),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(equity),
        "hit_rate": hit_rate(returns),
        "days": int(len(returns)),
    }
    if trades is not None:
        out["trades"] = trades
    return out
