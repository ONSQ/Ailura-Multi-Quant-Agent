"""Technical indicators and a compact feature snapshot (pure pandas/numpy)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # avg_down == 0 (only gains) -> RSI 100; keep NaN during warm-up
    return out.where(avg_down != 0, 100.0).where(avg_up.notna() & avg_down.notna())


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": mid + k * sd, "bb_lower": mid - k * sd, "bb_width": (2 * k * sd) / mid})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def vwma(df: pd.DataFrame, n: int = 20) -> pd.Series:
    pv = (df["close"] * df["volume"]).rolling(n, min_periods=n).sum()
    v = df["volume"].rolling(n, min_periods=n).sum()
    return pv / v


def kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    low_n = df["low"].rolling(n, min_periods=n).min()
    high_n = df["high"].rolling(n, min_periods=n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / m1, adjust=False, min_periods=1).mean()
    d = k.ewm(alpha=1 / m2, adjust=False, min_periods=1).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"k": k, "d": d, "j": j})


def zscore(close: pd.Series, n: int = 20) -> pd.Series:
    m = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return (close - m) / sd.replace(0, np.nan)


def realized_vol(close: pd.Series, n: int = 20) -> pd.Series:
    return np.log(close).diff().rolling(n, min_periods=n).std(ddof=0) * np.sqrt(252)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with all indicator columns attached."""
    out = df.copy()
    c = out["close"]
    out["sma_10"] = sma(c, 10)
    out["sma_20"] = sma(c, 20)
    out["sma_50"] = sma(c, 50)
    out["sma_200"] = sma(c, 200)
    out["ema_10"] = ema(c, 10)
    out = out.join(macd(c))
    out["rsi_14"] = rsi(c)
    out = out.join(bollinger(c))
    out["atr_14"] = atr(out)
    out["vwma_20"] = vwma(out)
    kd = kdj(out)
    out["kdj_k"], out["kdj_d"], out["kdj_j"] = kd["k"], kd["d"], kd["j"]
    out["zscore_20"] = zscore(c)
    out["rvol_20"] = realized_vol(c)
    out["ret_1d"] = c.pct_change()
    out["ret_5d"] = c.pct_change(5)
    out["ret_20d"] = c.pct_change(20)
    out["ret_60d"] = c.pct_change(60)
    return out


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else round(v, 6)


def snapshot(df: pd.DataFrame) -> dict[str, Any]:
    """Latest-bar feature snapshot: indicators, regime flags and a technical score in [-1, 1]."""
    if df.empty:
        return {}
    e = enrich(df) if "rsi_14" not in df.columns else df
    last = e.iloc[-1]
    close = float(last["close"])
    feats: dict[str, Any] = {
        "close": round(close, 4),
        "as_of": str(e.index[-1].date()),
        "bars": int(len(e)),
    }
    for col in [
        "sma_10", "sma_20", "sma_50", "sma_200", "ema_10", "macd", "signal", "hist", "rsi_14",
        "bb_upper", "bb_lower", "bb_width", "atr_14", "vwma_20", "kdj_k", "kdj_d", "kdj_j",
        "zscore_20", "rvol_20", "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    ]:
        feats[col] = _f(last.get(col))

    # Regime / structure flags
    trend = 0.0
    if feats["sma_50"] and feats["sma_200"]:
        trend = 1.0 if feats["sma_50"] > feats["sma_200"] else -1.0
    feats["trend_regime"] = "uptrend" if trend > 0 else "downtrend" if trend < 0 else "unknown"
    rv = feats["rvol_20"]
    feats["vol_regime"] = "high" if rv is not None and rv > 0.35 else "low" if rv is not None and rv < 0.18 else "normal"
    if feats["atr_14"]:
        feats["atr_pct"] = round(feats["atr_14"] / close, 6)
    hi_60 = float(e["high"].tail(60).max())
    lo_60 = float(e["low"].tail(60).min())
    feats["support_60d"] = round(lo_60, 4)
    feats["resistance_60d"] = round(hi_60, 4)
    feats["pct_from_high_60d"] = round(close / hi_60 - 1, 6)

    # Composite technical score in [-1, 1], built to lead rather than lag.
    # Direction comes from intermediate-term momentum that SKIPS the latest month
    # (12-1 and 6-1, vol-scaled): the most robust trend signal in the literature,
    # and blind to the move that just happened. Short-term oscillators (RSI, MACD
    # histogram, 5-day return) only describe what already happened, so they are
    # used for timing: a sharp move in the signal's direction means "wait for a
    # better entry", not "chase".
    c = e["close"]
    rv = max(feats["rvol_20"] or float(np.log(c).diff().tail(60).std() * np.sqrt(252)) or 0.25, 0.05)  # floor: near-zero vol blows up the scaling

    def _mom(long: int, skip: int = 21) -> float | None:
        if len(c) <= long:
            return None
        r = float(c.iloc[-1 - skip] / c.iloc[-1 - long] - 1)
        return r / (rv * np.sqrt((long - skip) / 252))

    m12, m6 = _mom(252), _mom(126)
    feats["mom_12_1"] = _f(m12)
    feats["mom_6_1"] = _f(m6)
    parts = [float(np.clip(m / 1.5, -1, 1)) for m in (m12, m6) if m is not None]
    if not parts and feats["sma_50"] and feats["sma_200"]:
        parts.append(float(np.clip((feats["sma_50"] / feats["sma_200"] - 1) * 10, -1, 1)))
    direction = float(np.mean(parts)) if parts else 0.0

    # stretch of the last week in units of its own expected move
    stretch = 0.0
    if feats["ret_5d"] is not None and rv > 0:
        stretch = feats["ret_5d"] / (rv * np.sqrt(5 / 252))
    feats["stretch_5d"] = round(stretch, 4)
    chasing = np.sign(stretch) == np.sign(direction) and abs(stretch) > 1.5
    feats["entry_timing"] = "extended: wait for pullback" if chasing else "counter-trend move: better entry on the trend side" if np.sign(stretch) == -np.sign(direction) and abs(stretch) > 1.0 else "neutral"
    score = direction * (0.5 if chasing else 1.0)
    feats["technical_score"] = round(float(np.clip(score, -1, 1)), 4)
    return feats
