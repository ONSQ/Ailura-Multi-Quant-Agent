from datetime import date, timedelta

import pytest

from ailura.agents.schemas import FinalDecision
from ailura.config import AiluraConfig
from ailura.graph.orchestrator import TradingGraph, sanitize_decision
from ailura.graph.signal import rating_to_action
from ailura.graph.state import PortfolioContext
from ailura.llm.base import LLMRequest, LLMResponse
from ailura.memory.store import DecisionMemory


def test_full_pipeline_offline(graph, a_date):
    st = graph.run("AAPL", a_date)
    fd = st.final_decision
    assert fd is not None and fd.rating in ("Buy", "Overweight", "Hold", "Underweight", "Sell")
    assert fd.action == rating_to_action(fd.rating)
    assert set(st.reports) == {"market", "news", "sentiment", "fundamentals"}
    assert st.research_verdict is not None and st.trader_proposal is not None
    assert len(st.risk_debate) == 3 and st.debate_rounds_run >= 1
    assert st.ledger.calls == 4 + 2 * st.debate_rounds_run + 1 + 1 + 3 + 1
    assert (graph.config.results_dir / "AAPL" / f"{a_date}.json").exists()
    assert len(graph.memory.entries) == 1 and not graph.memory.entries[0].resolved


def test_pipeline_is_deterministic(config, provider, a_date):
    a = TradingGraph(config, provider=provider, memory=DecisionMemory(None)).run("GOOGL", a_date, save_log=False).final_decision
    b = TradingGraph(config, provider=provider, memory=DecisionMemory(None)).run("GOOGL", a_date, save_log=False).final_decision
    assert a == b


def test_no_lookahead_in_decision(config, provider, a_date):
    """The decision for a date must not depend on data after it."""
    from ailura.data.base import MarketDataProvider

    class Truncated:
        name = "trunc"

        def __init__(self, inner: MarketDataProvider, cutoff: date) -> None:
            self.inner, self.cutoff = inner, cutoff

        def get_ohlcv(self, t, s, e):
            return self.inner.get_ohlcv(t, s, min(e, self.cutoff))

        def get_news(self, *a, **k):
            return self.inner.get_news(*a, **k)

        def get_social(self, *a, **k):
            return self.inner.get_social(*a, **k)

        def get_fundamentals(self, *a, **k):
            return self.inner.get_fundamentals(*a, **k)

    full = TradingGraph(config, provider=provider, memory=DecisionMemory(None)).run("AMZN", a_date, save_log=False)
    trunc = TradingGraph(config, provider=Truncated(provider, a_date), memory=DecisionMemory(None)).run("AMZN", a_date, save_log=False)
    assert full.final_decision == trunc.final_decision
    assert full.features == trunc.features


def test_early_stop_debate(config, provider, a_date):
    cfg = config.model_copy(update={"max_debate_rounds": 4, "early_stop_debate": True})
    st = TradingGraph(cfg, provider=provider, memory=DecisionMemory(None)).run("AAPL", a_date, save_log=False)
    assert st.debate_stopped_early and st.debate_rounds_run < 4
    cfg2 = config.model_copy(update={"max_debate_rounds": 4, "early_stop_debate": False})
    st2 = TradingGraph(cfg2, provider=provider, memory=DecisionMemory(None)).run("AAPL", a_date, save_log=False)
    assert st2.debate_rounds_run == 4


def test_memory_resolves_point_in_time(config, provider, a_date):
    g = TradingGraph(config, provider=provider, memory=DecisionMemory(None))
    g.run("AAPL", a_date, save_log=False)
    assert g.memory.pending()
    # a run the next day must NOT resolve (window not closed)
    g.run("AAPL", a_date + timedelta(days=1), save_log=False)
    assert len(g.memory.pending()) == 2
    # a run 3 weeks later resolves the first two and can recall them
    st = g.run("AAPL", a_date + timedelta(days=21), save_log=False)
    assert len(g.memory.pending()) == 1
    assert g.memory.stats()["resolved"] == 2
    lessons = g.memory.recall("AAPL", a_date + timedelta(days=21), k=5)
    assert lessons and all(m["decision_date"] <= str(a_date + timedelta(days=1)) for m in lessons)
    assert st.memories == lessons[: config.memory_k]
    # recall at the original date sees nothing (nothing was resolved before it)
    assert g.memory.recall("AAPL", a_date) == []


def test_sanitize_decision_enforces_invariants():
    pf = PortfolioContext(max_position_fraction=0.5, allow_short=False)
    raw = FinalDecision(rating="Buy", action="SELL", conviction=0.8, position_fraction=-0.9, stop_loss=120, take_profit=90, executive_summary="s", thesis="t")
    d = sanitize_decision(raw, pf, close=100.0)
    assert d.action == "BUY" and d.position_fraction == 0.5 and d.stop_loss is None and d.take_profit is None
    raw2 = FinalDecision(rating="Sell", action="SELL", conviction=-0.8, position_fraction=-0.9, executive_summary="s", thesis="t")
    assert sanitize_decision(raw2, pf, 100.0).action == "HOLD"
    assert sanitize_decision(raw2, PortfolioContext(allow_short=True), 100.0).position_fraction == -0.9


def test_structured_output_failure_is_loud(config, provider, a_date):
    from ailura.llm.base import StructuredOutputError

    class Junk:
        name = "junk"

        def complete(self, req: LLMRequest) -> LLMResponse:
            return LLMResponse(text="not json", parsed=None, model="junk")

    g = TradingGraph(config, provider=provider, memory=DecisionMemory(None), quick_llm=Junk(), deep_llm=Junk())
    with pytest.raises(StructuredOutputError):
        g.run("AAPL", a_date, save_log=False)


def test_agent_strategy_backtest(config, provider):
    from ailura.backtest import AgentStrategy, Backtester

    g = TradingGraph(config, provider=provider, memory=DecisionMemory(None))
    prices = provider.get_ohlcv("AAPL", date(2024, 2, 1), date(2024, 2, 29))
    strat = AgentStrategy(g, "AAPL", rebalance_every=5)
    w = strat.signals(prices)
    res = Backtester().run(prices, w, "agent")
    assert len(strat.decisions) == (len(prices) + 4) // 5
    assert "sharpe" in res.metrics and (w.abs() <= 1).all()


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("AILURA_MAX_DEBATE_ROUNDS", "3")
    monkeypatch.setenv("AILURA_EARLY_STOP_DEBATE", "false")
    monkeypatch.setenv("AILURA_ANALYSTS", "market,news")
    cfg = AiluraConfig.from_env()
    assert cfg.max_debate_rounds == 3 and cfg.early_stop_debate is False and cfg.analysts == ["market", "news"]
    monkeypatch.setenv("AILURA_EARLY_STOP_DEBATE", "maybe")
    with pytest.raises(ValueError):
        AiluraConfig.from_env()
