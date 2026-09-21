"""TradingGraph: the end-to-end multi-agent decision pipeline.

Order (mirrors the paper): analysts -> bull/bear debate -> research manager ->
trader -> risk debate (aggressive / conservative / neutral) -> portfolio manager.

Improvements over the reference implementation:
* point-in-time data guard on every input;
* pre-fetched inputs instead of per-analyst tool loops;
* early-stopped debates on convergence (token-overlap novelty);
* structured outputs with a deterministic post-check (action <-> rating,
  position limits, stop placement);
* outcome-based memory resolved *as of* the decision date, so backtests learn
  only from the past.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from datetime import date

from ailura.agents.analysts import ANALYSTS
from ailura.agents.researchers import BearResearcher, BullResearcher, ResearchManager
from ailura.agents.risk import RISK_DEBATORS, PortfolioManager
from ailura.agents.schemas import FinalDecision
from ailura.agents.trader import Trader
from ailura.config import AiluraConfig
from ailura.data.base import MarketDataProvider, PointInTimeProvider, to_date
from ailura.data.factory import build_provider
from ailura.data.indicators import snapshot
from ailura.graph.signal import rating_to_action
from ailura.graph.state import PortfolioContext, TradingState
from ailura.llm.base import LLMClient
from ailura.llm.factory import build_llm
from ailura.memory.reflection import Reflector
from ailura.memory.store import DecisionMemory

StepHook = Callable[[str, TradingState], None]
_TOKEN = re.compile(r"[a-z0-9]{3,}")


def _tokens(*texts: str) -> set[str]:
    return set(_TOKEN.findall(" ".join(texts).lower()))


def sanitize_decision(d: FinalDecision, portfolio: PortfolioContext, close: float | None) -> FinalDecision:
    """Enforce invariants the model may violate: action<->rating, limits, stop side."""
    action = rating_to_action(d.rating)
    frac = d.position_fraction
    if action == "HOLD":
        frac = 0.0
    elif action == "BUY":
        frac = abs(frac)
    else:
        frac = -abs(frac)
    if not portfolio.allow_short and frac < 0:
        frac = 0.0
        action = "HOLD"
    frac = max(-portfolio.max_position_fraction, min(portfolio.max_position_fraction, frac))
    stop, tp = d.stop_loss, d.take_profit
    if close is not None:
        if action == "BUY":
            if stop is not None and stop >= close:
                stop = None
            if tp is not None and tp <= close:
                tp = None
        elif action == "SELL":
            if stop is not None and stop <= close:
                stop = None
            if tp is not None and tp >= close:
                tp = None
        else:
            stop = tp = None
    return d.model_copy(update={"action": action, "position_fraction": round(frac, 4), "stop_loss": stop, "take_profit": tp})


class TradingGraph:
    def __init__(
        self,
        config: AiluraConfig | None = None,
        *,
        quick_llm: LLMClient | None = None,
        deep_llm: LLMClient | None = None,
        provider: MarketDataProvider | None = None,
        memory: DecisionMemory | None = None,
        on_step: StepHook | None = None,
    ) -> None:
        self.config = config or AiluraConfig()
        self.quick_llm = quick_llm or build_llm(self.config, "quick")
        self.deep_llm = deep_llm or build_llm(self.config, "deep")
        self.provider = provider or build_provider(self.config)
        self.memory = memory if memory is not None else DecisionMemory(self.config.memory_path)
        self.reflector = Reflector(self.quick_llm)
        self.on_step = on_step

        self.analysts = [ANALYSTS[a](self.quick_llm) for a in self.config.analysts]
        self.bull = BullResearcher(self.quick_llm)
        self.bear = BearResearcher(self.quick_llm)
        self.research_manager = ResearchManager(self.deep_llm)
        self.trader = Trader(self.quick_llm)
        self.risk_team = [cls(self.quick_llm) for cls in RISK_DEBATORS]
        self.portfolio_manager = PortfolioManager(self.deep_llm)

    # ------------------------------------------------------------- helpers
    def _step(self, name: str, state: TradingState, t0: float) -> None:
        state.timings[name] = round(time.perf_counter() - t0, 4)
        if self.on_step:
            self.on_step(name, state)

    def prepare_state(self, ticker: str, trade_date: date, portfolio: PortfolioContext | None = None) -> TradingState:
        trade_date = to_date(trade_date)
        pit = PointInTimeProvider(self.provider, trade_date, strict=True)
        hist = pit.history(ticker, self.config.lookback_days)
        if hist.empty:
            raise ValueError(f"no price history for {ticker} as of {trade_date}")
        feats = snapshot(hist)
        bars = [
            {"date": str(ix.date()), **{c: round(float(row[c]), 4) for c in ("open", "high", "low", "close")}, "volume": float(row["volume"])}
            for ix, row in hist.tail(15).iterrows()
        ]
        news = [n.__dict__ | {"published": str(n.published)} for n in pit.get_news(ticker, lookback_days=self.config.news_lookback_days)]
        social = [p.__dict__ | {"posted": str(p.posted)} for p in pit.get_social(ticker, lookback_days=self.config.news_lookback_days)]
        fund = pit.get_fundamentals(ticker)
        pf = portfolio or PortfolioContext(
            max_position_fraction=self.config.max_position_fraction,
            allow_short=self.config.allow_short,
            target_annual_vol=self.config.target_annual_vol,
        )
        tags = [str(feats.get("trend_regime")), f"{feats.get('vol_regime')}-vol"]
        memories = self.memory.recall(ticker, trade_date, k=self.config.memory_k, tags=tags)
        return TradingState(
            ticker=ticker.upper(), trade_date=trade_date, features=feats, recent_bars=bars,
            news=news, social=social,
            fundamentals={"as_of": str(fund.as_of), "metrics": fund.metrics, "notes": fund.notes},
            memories=memories, portfolio=pf,
        )

    # ---------------------------------------------------------------- phases
    def _debate(self, state: TradingState) -> None:
        seen: set[str] = set()
        for rnd in range(self.config.max_debate_rounds):
            new_round: set[str] = set()
            any_points = False
            for agent in (self.bull, self.bear):
                arg = agent.run(state)
                state.debate.append(arg)
                any_points = any_points or bool(arg.points)
                new_round |= _tokens(*arg.points, *arg.rebuttals)
            state.debate_rounds_run = rnd + 1
            if not self.config.early_stop_debate or rnd == 0:
                seen |= new_round
                continue
            # converged when neither side adds evidence, or when this round's evidence is mostly a repeat
            novelty = len(new_round - seen) / len(new_round) if new_round else 0.0
            seen |= new_round
            if not any_points or novelty < self.config.debate_novelty_threshold:
                state.debate_stopped_early = True
                break

    def _risk(self, state: TradingState) -> None:
        for rnd in range(self.config.max_risk_rounds):
            for agent in self.risk_team:
                state.risk_debate.append(agent.run(state))
            state.risk_rounds_run = rnd + 1

    # ------------------------------------------------------------------ run
    def run(
        self,
        ticker: str,
        trade_date: date,
        portfolio: PortfolioContext | None = None,
        *,
        record: bool = True,
        resolve_memory: bool = True,
        save_log: bool = True,
    ) -> TradingState:
        trade_date = to_date(trade_date)
        if resolve_memory:
            # settle past decisions whose windows closed before this date (point-in-time)
            self.memory.resolve_pending(self.provider, self.reflector, benchmark=self.config.benchmark_ticker, as_of=trade_date)
        t0 = time.perf_counter()
        state = self.prepare_state(ticker, trade_date, portfolio)
        self._step("prepare", state, t0)

        for agent in self.analysts:
            t = time.perf_counter()
            rep = agent.run(state)
            state.reports[rep.analyst] = rep
            self._step(agent.role, state, t)

        t = time.perf_counter()
        self._debate(state)
        self._step("debate", state, t)
        t = time.perf_counter()
        state.research_verdict = self.research_manager.run(state)
        self._step("research_manager", state, t)
        t = time.perf_counter()
        state.trader_proposal = self.trader.run(state)
        self._step("trader", state, t)
        t = time.perf_counter()
        self._risk(state)
        self._step("risk_debate", state, t)
        t = time.perf_counter()
        raw = self.portfolio_manager.run(state)
        state.final_decision = sanitize_decision(raw, state.portfolio, state.close)
        self._step("portfolio_manager", state, t)
        state.timings["total"] = round(time.perf_counter() - t0, 4)

        if record and state.final_decision is not None:
            fd = state.final_decision
            self.memory.record(
                ticker=state.ticker, decision_date=trade_date, rating=fd.rating, conviction=fd.conviction,
                position_fraction=fd.position_fraction, executive_summary=fd.executive_summary,
                thesis=fd.thesis, holding_days=self.config.holding_days,
                tags=[str(state.features.get("trend_regime")), f"{state.features.get('vol_regime')}-vol"],
            )
        if save_log:
            self._save(state)
        return state

    def decide(self, ticker: str, trade_date: date, **kw) -> FinalDecision:
        st = self.run(ticker, trade_date, **kw)
        assert st.final_decision is not None
        return st.final_decision

    def _save(self, state: TradingState) -> None:
        out = self.config.results_dir / state.ticker
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{state.trade_date}.json").write_text(json.dumps(state.to_record(), indent=1, default=str))
