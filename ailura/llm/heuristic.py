"""HeuristicLLM: a deterministic, network-free stand-in for a language model.

It dispatches on ``request.role`` and produces schema-valid outputs from the
structured ``request.context`` using transparent quant rules. Two purposes:

1. Tests and CI run the *entire* multi-agent pipeline offline.
2. It doubles as a non-LLM "rules ensemble" baseline, so the marginal value
   of a real LLM can be measured against an identical pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np

from ailura.agents.schemas import (
    AnalystReport,
    DebateArgument,
    FinalDecision,
    ReflectionLesson,
    ResearchVerdict,
    RiskArgument,
    Signal,
    TraderProposal,
)
from ailura.graph.signal import conviction_to_rating, rating_to_action
from ailura.llm.base import LLMRequest, LLMResponse, TimedCall, Usage

_ANALYST_WEIGHTS = {"market": 1.0, "fundamentals": 0.8, "news": 0.7, "sentiment": 0.5}


def _stance(score: float, band: float = 0.15) -> str:
    return "bullish" if score > band else "bearish" if score < -band else "neutral"


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


class HeuristicLLM:
    name = "heuristic"

    def __init__(self) -> None:
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "market_analyst": self.market_analyst,
            "news_analyst": self.news_analyst,
            "sentiment_analyst": self.sentiment_analyst,
            "fundamentals_analyst": self.fundamentals_analyst,
            "bull_researcher": lambda c: self.researcher(c, "bull"),
            "bear_researcher": lambda c: self.researcher(c, "bear"),
            "research_manager": self.research_manager,
            "trader": self.trader,
            "aggressive_risk": lambda c: self.risk(c, "aggressive"),
            "conservative_risk": lambda c: self.risk(c, "conservative"),
            "neutral_risk": lambda c: self.risk(c, "neutral"),
            "portfolio_manager": self.portfolio_manager,
            "reflector": self.reflector,
        }

    def complete(self, request: LLMRequest) -> LLMResponse:
        handler = self.handlers.get(request.role)
        if handler is None:
            raise KeyError(f"HeuristicLLM has no handler for role {request.role!r}")
        with TimedCall() as t:
            out = handler(request.context)
        text = json.dumps(out.model_dump(), default=str)
        return LLMResponse(
            text=text,
            parsed=out,
            usage=Usage(input_tokens=(len(request.system) + len(request.user)) // 4, output_tokens=len(text) // 4),
            model="heuristic",
            latency_s=t.elapsed,
            stop_reason="end_turn",
        )

    # ------------------------------------------------------------ analysts
    def market_analyst(self, c: dict[str, Any]) -> AnalystReport:
        f = c.get("features", {})
        score = float(f.get("technical_score") or 0.0)
        sig: list[Signal] = []
        if f.get("rsi_14") is not None:
            r = f["rsi_14"]
            sig.append(Signal(name="RSI(14)", value=f"{r:.1f}", interpretation="overbought" if r > 70 else "oversold" if r < 30 else "neutral momentum", direction="bearish" if r > 70 else "bullish" if r < 30 else "neutral"))
        if f.get("hist") is not None:
            h = f["hist"]
            sig.append(Signal(name="MACD histogram", value=f"{h:.3f}", interpretation="momentum expanding" if h > 0 else "momentum contracting", direction="bullish" if h > 0 else "bearish"))
        if f.get("sma_50") and f.get("sma_200"):
            up = f["sma_50"] > f["sma_200"]
            sig.append(Signal(name="SMA50 vs SMA200", value=f"{f['sma_50']:.2f} / {f['sma_200']:.2f}", interpretation="golden-cross regime" if up else "death-cross regime", direction="bullish" if up else "bearish"))
        if f.get("bb_width") is not None:
            sig.append(Signal(name="Bollinger width", value=f"{f['bb_width']:.3f}", interpretation="volatility " + ("expansion" if f["bb_width"] > 0.12 else "compression")))
        if f.get("atr_pct") is not None:
            sig.append(Signal(name="ATR% of price", value=f"{f['atr_pct']:.2%}", interpretation=f"vol regime {f.get('vol_regime')}"))
        if f.get("zscore_20") is not None:
            z = f["zscore_20"]
            sig.append(Signal(name="20d z-score", value=f"{z:.2f}", interpretation="stretched above mean" if z > 1.5 else "stretched below mean" if z < -1.5 else "near mean", direction="bearish" if z > 1.5 else "bullish" if z < -1.5 else "neutral"))
        bars = int(f.get("bars") or 0)
        conf = _clip(0.35 + 0.5 * min(1.0, bars / 120) , 0, 1)
        summary = (
            f"Price {f.get('close')} in a {f.get('trend_regime', 'unknown')} with {f.get('vol_regime', 'normal')} volatility. "
            f"Composite technical score {score:+.2f}. Support {f.get('support_60d')} / resistance {f.get('resistance_60d')}; "
            f"20-day return {f.get('ret_20d')}, 60-day {f.get('ret_60d')}."
        )
        return AnalystReport(analyst="market", summary=summary, signals=sig, stance=_stance(score), score=_clip(score), confidence=conf, data_quality=1.0 if bars >= 60 else max(0.2, bars / 60))

    def news_analyst(self, c: dict[str, Any]) -> AnalystReport:
        s = c.get("avg_sentiment")
        n = int(c.get("news_count") or 0)
        score = _clip(float(s) if s is not None else 0.0)
        sig = [Signal(name="headline count", value=str(n), interpretation="news flow intensity")]
        if s is not None:
            sig.append(Signal(name="avg headline sentiment", value=f"{s:+.2f}", interpretation="net positive coverage" if s > 0.15 else "net negative coverage" if s < -0.15 else "mixed coverage", direction=_stance(score)))
        for item in (c.get("news") or [])[:3]:
            sig.append(Signal(name="headline", value=item.get("headline", "")[:80], interpretation=item.get("summary", "")[:100], direction=_stance(float(item.get("sentiment") or 0))))
        dq = 1.0 if n >= 3 else n / 3
        return AnalystReport(analyst="news", summary=f"{n} headlines in window; average sentiment {s}. Coverage is {_stance(score)}.", signals=sig, stance=_stance(score), score=score, confidence=_clip(0.3 + 0.4 * dq, 0, 1), data_quality=dq)

    def sentiment_analyst(self, c: dict[str, Any]) -> AnalystReport:
        w = c.get("engagement_weighted_sentiment")
        n = int(c.get("post_count") or 0)
        r5 = c.get("ret_5d")
        raw = float(w) if w is not None else 0.0
        # crowd extremes after a big move are treated as contrarian
        contrarian = r5 is not None and abs(raw) > 0.6 and np.sign(raw) == np.sign(r5) and abs(r5) > 0.05
        score = _clip(-0.5 * raw if contrarian else 0.7 * raw)
        sig = [Signal(name="posts", value=str(n), interpretation="discussion volume"),
               Signal(name="engagement-weighted sentiment", value=f"{raw:+.2f}", interpretation="crowd euphoric, contrarian fade" if contrarian and raw > 0 else "crowd capitulating, contrarian buy" if contrarian else "crowd mood", direction=_stance(score))]
        dq = 1.0 if n >= 10 else n / 10
        return AnalystReport(analyst="sentiment", summary=f"{n} posts; engagement-weighted sentiment {raw:+.2f}; {'contrarian' if contrarian else 'confirming'} read vs 5-day return {r5}.", signals=sig, stance=_stance(score), score=score, confidence=_clip(0.25 + 0.35 * dq, 0, 1), data_quality=dq)

    def fundamentals_analyst(self, c: dict[str, Any]) -> AnalystReport:
        m = (c.get("fundamentals") or {}).get("metrics") or {}
        if not m:
            return AnalystReport(analyst="fundamentals", summary="No fundamentals available as of this date.", stance="neutral", score=0.0, confidence=0.1, data_quality=0.0)
        parts: list[float] = []
        sig: list[Signal] = []
        if "eps_growth_yoy" in m:
            g = m["eps_growth_yoy"]
            parts.append(_clip(g / 0.25))
            sig.append(Signal(name="EPS growth YoY", value=f"{g:+.1%}", interpretation="strong earnings growth" if g > 0.15 else "weak earnings growth" if g < 0 else "moderate growth", direction=_stance(g / 0.25)))
        if "revenue_growth_yoy" in m:
            g = m["revenue_growth_yoy"]
            parts.append(_clip(g / 0.2))
            sig.append(Signal(name="Revenue growth YoY", value=f"{g:+.1%}", interpretation="top-line momentum", direction=_stance(g / 0.2)))
        if "pe_ratio" in m:
            pe = m["pe_ratio"]
            v = _clip((25 - pe) / 20)
            parts.append(v)
            sig.append(Signal(name="P/E", value=f"{pe:.1f}", interpretation="rich valuation" if pe > 35 else "cheap valuation" if pe < 15 else "fair valuation", direction=_stance(v)))
        if "free_cash_flow_yield" in m:
            y = m["free_cash_flow_yield"]
            v = _clip((y - 0.03) / 0.03)
            parts.append(v)
            sig.append(Signal(name="FCF yield", value=f"{y:.1%}", interpretation="cash generation vs price", direction=_stance(v)))
        if "net_debt_to_ebitda" in m:
            d = m["net_debt_to_ebitda"]
            v = _clip((1.5 - d) / 2)
            parts.append(0.5 * v)
            sig.append(Signal(name="Net debt / EBITDA", value=f"{d:.1f}x", interpretation="leverage", direction=_stance(v)))
        score = _clip(float(np.mean(parts)) if parts else 0.0)
        as_of = (c.get("fundamentals") or {}).get("as_of")
        return AnalystReport(analyst="fundamentals", summary=f"Latest filing as of {as_of}: composite fundamental score {score:+.2f} across growth, valuation, cash flow and leverage.", signals=sig, stance=_stance(score), score=score, confidence=0.6, data_quality=1.0)

    # ---------------------------------------------------------- researchers
    def researcher(self, c: dict[str, Any], side: str) -> DebateArgument:
        want = "bullish" if side == "bull" else "bearish"
        rnd = int(c.get("round") or 1)
        reports = c.get("reports") or []
        supporting: list[tuple[float, str]] = []
        for r in reports:
            for s in r.get("signals", []):
                if s.get("direction") == want:
                    supporting.append((r.get("confidence", 0.5), f"[{r['analyst']}] {s['name']} = {s['value']}: {s['interpretation']}"))
            if r.get("stance") == want:
                supporting.append((r.get("confidence", 0.5) + 0.5, f"[{r['analyst']}] overall {want} ({r.get('score'):+.2f}): {r.get('summary')}"))
        supporting.sort(key=lambda t: t[0], reverse=True)
        # round 1: strongest half; round 2: the rest; round 3+: nothing new (converged)
        n = len(supporting)
        if rnd == 1:
            chosen = supporting[: max(1, (n + 1) // 2)]
        elif rnd == 2:
            chosen = supporting[(n + 1) // 2 :] or supporting[:1]
        else:
            chosen = []
        points = [p for _, p in chosen][:8]
        opp = c.get("opponent_latest") or {}
        rebuttals = [f"'{p[:70]}' is already reflected in price and does not change the {side} thesis" for p in (opp.get("points") or [])[:3]]
        conceded = [p for p in (opp.get("points") or [])[3:4]]
        aligned = [r for r in reports if r.get("stance") == want]
        conf = _clip(0.3 + 0.15 * len(aligned) + 0.05 * len(points), 0, 0.95)
        thesis = (f"The {side} case rests on {len(aligned)} of {len(reports)} analysts leaning {want}." if points else f"No new {side} evidence this round; the debate has converged.")
        return DebateArgument(side=side, thesis=thesis, points=points, rebuttals=rebuttals if points else [], conceded=conceded, confidence=conf if points else 0.2)

    def research_manager(self, c: dict[str, Any]) -> ResearchVerdict:
        reports = c.get("reports") or []
        num = den = 0.0
        for r in reports:
            w = _ANALYST_WEIGHTS.get(r["analyst"], 0.5) * r.get("confidence", 0.5) * r.get("data_quality", 1.0)
            num += w * r.get("score", 0.0)
            den += w
        conviction = _clip(num / den if den else 0.0)
        # apply lessons: shrink conviction when past calls in a similar setup were wrong
        wrong = [m for m in (c.get("past_lessons") or []) if m.get("verdict") == "incorrect"]
        if wrong:
            conviction *= 0.8 ** min(len(wrong), 3)
        bull = [a for a in (c.get("debate") or []) if a["side"] == "bull"]
        bear = [a for a in (c.get("debate") or []) if a["side"] == "bear"]
        bull_c = max((a["confidence"] for a in bull), default=0.0)
        bear_c = max((a["confidence"] for a in bear), default=0.0)
        conviction = _clip(0.8 * conviction + 0.2 * (bull_c - bear_c))
        rating = conviction_to_rating(conviction)
        decisive = [p for a in (bull if conviction >= 0 else bear) for p in a["points"]][:4]
        risks = [p for a in (bear if conviction >= 0 else bull) for p in a["points"]][:4]
        return ResearchVerdict(rating=rating, conviction=round(conviction, 4), rationale=f"Confidence-weighted analyst score {conviction:+.2f}; bull confidence {bull_c:.2f} vs bear {bear_c:.2f}; {len(wrong)} adverse past lessons applied.", decisive_arguments=decisive, key_risks=risks, plan=f"Size at {abs(conviction):.0%} of standard; re-evaluate on regime change or thesis invalidation.")

    # --------------------------------------------------------------- trader
    def trader(self, c: dict[str, Any]) -> TraderProposal:
        v = c.get("verdict") or {}
        p = c.get("price") or {}
        pf = c.get("portfolio") or {}
        conv = float(v.get("conviction", 0.0))
        rating = v.get("rating", "Hold")
        action = rating_to_action(rating)
        close = p.get("close")
        atr = p.get("atr_14") or (close * 0.02 if close else None)
        rvol = p.get("rvol_20") or 0.25
        target = float(pf.get("target_annual_vol", 0.2))
        vol_scalar = min(1.0, target / max(rvol, 1e-6))
        frac = _clip(conv * vol_scalar, -float(pf.get("max_position_fraction", 1.0)), float(pf.get("max_position_fraction", 1.0)))
        if not pf.get("allow_short", False):
            frac = max(0.0, frac)
        if action == "HOLD":
            frac = 0.0
        long = action == "BUY"
        stop = tp = None
        if close and atr:
            stop = round(close - 2 * atr, 4) if long else round(close + 2 * atr, 4) if action == "SELL" else None
            tp = round(close + 3 * atr, 4) if long else round(close - 3 * atr, 4) if action == "SELL" else None
        return TraderProposal(action=action, rating=rating, conviction=round(conv, 4), entry_price=close, stop_loss=stop, take_profit=tp, position_fraction=round(frac, 4), horizon_days=5, reasoning=f"{rating} from research; vol scalar {vol_scalar:.2f} (target {target:.0%} / realized {rvol:.0%}); stops at 2 ATR, target 3 ATR.")

    # ----------------------------------------------------------------- risk
    def risk(self, c: dict[str, Any], persona: str) -> RiskArgument:
        prop = c.get("proposal") or {}
        r = c.get("risk") or {}
        pf = c.get("portfolio") or {}
        base = float(prop.get("position_fraction", 0.0))
        max_pos = float(pf.get("max_position_fraction", 1.0))
        close, atr_pct = r.get("close"), r.get("atr_pct") or 0.02
        high_vol = r.get("vol_regime") == "high"
        if persona == "aggressive":
            rec = _clip(base * 1.3, -max_pos, max_pos)
            stance = "scale_up" if abs(base) > 0.05 else "support"
            pts = [f"Edge is {prop.get('conviction'):+.2f}; under-sizing wastes it", "Trend regime favours momentum continuation"]
            stop = prop.get("stop_loss")
        elif persona == "conservative":
            rec = _clip(base * (0.5 if high_vol else 0.7), -max_pos, max_pos)
            stance = "scale_down" if abs(base) > 0.05 else "support"
            pts = [f"ATR is {atr_pct:.1%} of price; a 2-ATR stop risks {2 * atr_pct:.1%}", f"Drawdown from 60d high {r.get('pct_from_high_60d')}", "Thesis relies on analysts with partial data" if high_vol else "Size for the possibility the thesis is wrong"]
            stop = round(close * (1 - 1.5 * atr_pct), 4) if close and base > 0 else round(close * (1 + 1.5 * atr_pct), 4) if close and base < 0 else None
        else:
            rec = _clip(base * (0.85 if high_vol else 1.0), -max_pos, max_pos)
            stance = "support"
            pts = ["Vol-targeted size already embedded in proposal", "Aggressive overstates edge persistence; conservative overstates tail risk"]
            stop = prop.get("stop_loss")
        rebut = [f"{o['persona']} recommends {o['recommended_position_fraction']:+.2f}; " + ("too timid given the edge" if persona == "aggressive" else "ignores vol regime" if persona == "conservative" else "both are within a reasonable band") for o in (c.get("others_latest") or [])]
        if not pf.get("allow_short", False):
            rec = max(0.0, rec)
        return RiskArgument(persona=persona, stance_on_proposal=stance, points=pts, rebuttals=rebut, recommended_position_fraction=round(rec, 4), recommended_stop_loss=stop, confidence=0.6)

    def portfolio_manager(self, c: dict[str, Any]) -> FinalDecision:
        prop = c.get("proposal") or {}
        v = c.get("verdict") or {}
        pf = c.get("portfolio") or {}
        f = c.get("features") or {}
        recs = {a["persona"]: a for a in (c.get("risk_debate") or [])}
        w = {"neutral": 0.5, "aggressive": 0.25, "conservative": 0.25}
        num = sum(w[p] * recs[p]["recommended_position_fraction"] for p in recs if p in w)
        den = sum(w[p] for p in recs if p in w)
        frac = num / den if den else float(prop.get("position_fraction", 0.0))
        max_pos = float(pf.get("max_position_fraction", 1.0))
        frac = _clip(frac, -max_pos, max_pos)
        if not pf.get("allow_short", False):
            frac = max(0.0, frac)
        conv = float(v.get("conviction", prop.get("conviction", 0.0)))
        rating = conviction_to_rating(conv)
        action = rating_to_action(rating)
        if action == "HOLD":
            frac = 0.0
        cons = recs.get("conservative", {})
        stop = cons.get("recommended_stop_loss") if f.get("vol_regime") == "high" else prop.get("stop_loss")
        lessons = [m.get("lesson", "") for m in (c.get("past_lessons") or [])][:3]
        return FinalDecision(
            rating=rating, action=action, conviction=round(conv, 4), position_fraction=round(frac, 4),
            stop_loss=stop, take_profit=prop.get("take_profit"),
            executive_summary=f"{action} {c.get('ticker')} at {frac:+.0%} of max allocation ({rating}, conviction {conv:+.2f}). Risk team consensus {num / den if den else frac:+.2f} after a {f.get('vol_regime')}-vol, {f.get('trend_regime')} read.",
            thesis=v.get("rationale", ""),
            invalidation_triggers=[f"Close beyond stop {stop}", "Trend regime flips", "Analyst data quality drops"],
            lessons_applied=[text for text in lessons if text],
        )

    # ------------------------------------------------------------ reflector
    def reflector(self, c: dict[str, Any]) -> ReflectionLesson:
        out = c.get("outcome") or {}
        alpha = float(out.get("alpha", 0.0))
        rating = c.get("rating", "Hold")
        directional = rating in ("Buy", "Overweight", "Sell", "Underweight")
        expected = 1 if rating in ("Buy", "Overweight") else -1 if rating in ("Sell", "Underweight") else 0
        if not directional:
            verdict = "inconclusive"
        elif abs(alpha) < 0.005:
            verdict = "inconclusive"
        else:
            verdict = "correct" if np.sign(alpha) == expected else "incorrect"
        hd = c.get("holding_days", 5)
        lesson = (
            f"{rating} calls in this setup earned {alpha:+.1%} alpha over {hd} days; keep sizing." if verdict == "correct"
            else f"{rating} in this setup lost {alpha:+.1%} alpha over {hd} days; require confirmation before sizing up." if verdict == "incorrect"
            else f"{hd}-day window ({alpha:+.1%} alpha) too small to judge; extend horizon before drawing conclusions."
        )
        return ReflectionLesson(verdict=verdict, what_the_outcome_shows=f"Raw {out.get('raw_return')}, benchmark {out.get('benchmark_return')}, alpha {alpha:+.2%}.", thesis_component_affected="directional call", lesson=lesson, tags=list(c.get("tags") or [])[:6])
