"""Ailura command-line interface."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from ailura.config import AiluraConfig

app = typer.Typer(help="Ailura Multi-Quant Agent: multi-agent LLM trading research.", no_args_is_help=True)
console = Console()


def _config(provider: str | None, llm: str | None, **kw) -> AiluraConfig:
    overrides = {k: v for k, v in kw.items() if v is not None}
    if provider:
        overrides["data_provider"] = provider
    if llm:
        overrides["llm_provider"] = llm
    return AiluraConfig.from_env(**overrides)


@app.command()
def analyze(
    ticker: str,
    date_: str = typer.Option(None, "--date", help="Decision date YYYY-MM-DD (default: today)"),
    provider: str = typer.Option(None, help="synthetic | csv | yfinance"),
    llm: str = typer.Option(None, help="heuristic | anthropic | openai"),
    debate_rounds: int = typer.Option(None, help="max bull/bear rounds"),
    json_out: bool = typer.Option(False, "--json", help="print the full state as JSON"),
) -> None:
    """Run the full multi-agent pipeline for one ticker and date."""
    from ailura.graph.orchestrator import TradingGraph

    cfg = _config(provider, llm, max_debate_rounds=debate_rounds)
    d = date.fromisoformat(date_) if date_ else date.today()

    def hook(name: str, state) -> None:
        console.print(f"[dim]✓ {name} ({state.timings.get(name, 0):.2f}s)[/dim]")

    graph = TradingGraph(cfg, on_step=hook)
    state = graph.run(ticker, d)
    fd = state.final_decision
    assert fd is not None
    if json_out:
        console.print_json(json.dumps(state.to_record(), default=str))
        return
    for name, rep in state.reports.items():
        console.print(Panel(f"[b]{rep.stance}[/b] score {rep.score:+.2f} conf {rep.confidence:.2f}\n{rep.summary}", title=f"{name} analyst"))
    if state.research_verdict:
        v = state.research_verdict
        console.print(Panel(f"[b]{v.rating}[/b] conviction {v.conviction:+.2f} (debate rounds {state.debate_rounds_run}{', early stop' if state.debate_stopped_early else ''})\n{v.rationale}", title="research manager"))
    color = {"BUY": "green", "SELL": "red"}.get(fd.action, "yellow")
    console.print(Panel(
        f"[{color} b]{fd.action}[/] {fd.rating}  conviction {fd.conviction:+.2f}  size {fd.position_fraction:+.0%}\n"
        f"stop {fd.stop_loss}  target {fd.take_profit}\n\n{fd.executive_summary}\n\n[dim]cost: {state.ledger.summary()}[/dim]",
        title=f"FINAL DECISION {state.ticker} {state.trade_date}",
    ))


@app.command()
def backtest(
    ticker: str,
    start: str = typer.Option(..., help="YYYY-MM-DD"),
    end: str = typer.Option(..., help="YYYY-MM-DD"),
    provider: str = typer.Option(None),
    llm: str = typer.Option(None),
    rebalance_every: int = typer.Option(1, help="run the agents every N trading days"),
    sizing: str = typer.Option("conviction", help="conviction | binary"),
    with_baselines: bool = typer.Option(True, help="also run the paper's baselines"),
    out: Path = typer.Option(None, help="write results JSON here"),
) -> None:
    """Backtest the agent pipeline (and baselines) over a date range."""
    from ailura.backtest import BASELINES, AgentStrategy, Backtester, ExecutionConfig
    from ailura.backtest.report import rich_table
    from ailura.data.base import normalize_ohlcv
    from ailura.graph.orchestrator import TradingGraph
    from ailura.memory.store import DecisionMemory

    cfg = _config(provider, llm)
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    graph = TradingGraph(cfg, memory=DecisionMemory(None))  # fresh in-memory lessons per backtest
    prices = normalize_ohlcv(graph.provider.get_ohlcv(ticker, s, e))
    if prices.empty:
        raise typer.BadParameter(f"no prices for {ticker} in {start}..{end}")
    bt = Backtester(ExecutionConfig(commission_bps=cfg.commission_bps, slippage_bps=cfg.slippage_bps, allow_short=cfg.allow_short))
    results = []
    if with_baselines:
        # baselines get a warm-up window so indicators are defined on day one
        warm = normalize_ohlcv(graph.provider.get_ohlcv(ticker, s - timedelta(days=400), e))
        for name, strat in BASELINES.items():
            w = strat.signals(warm).loc[prices.index]
            results.append(bt.run(prices, w, name))
    strat = AgentStrategy(graph, ticker, rebalance_every=rebalance_every, sizing=sizing,
                          on_decision=lambda st: console.print(f"[dim]{st.trade_date} {st.final_decision.action:<4} {st.final_decision.rating:<11} size {st.final_decision.position_fraction:+.2f}[/dim]"))
    results.append(bt.run(prices, strat.signals(prices), strat.name))
    console.print(rich_table(results, title=f"{ticker} {start} → {end}"))
    console.print(f"memory: {graph.memory.stats()}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"ticker": ticker, "start": start, "end": end, "results": [r.to_dict() for r in results], "decisions": strat.decisions}, indent=1, default=str))
        console.print(f"wrote {out}")


@app.command()
def baselines(
    ticker: str,
    start: str = typer.Option(...),
    end: str = typer.Option(...),
    provider: str = typer.Option(None),
) -> None:
    """Run only the paper's five baseline strategies."""
    from ailura.backtest import BASELINES, Backtester, ExecutionConfig
    from ailura.backtest.report import rich_table
    from ailura.data.base import normalize_ohlcv
    from ailura.data.factory import build_provider

    cfg = _config(provider, None)
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    prov = build_provider(cfg)
    prices = normalize_ohlcv(prov.get_ohlcv(ticker, s, e))
    warm = normalize_ohlcv(prov.get_ohlcv(ticker, s - timedelta(days=400), e))
    bt = Backtester(ExecutionConfig(commission_bps=cfg.commission_bps, slippage_bps=cfg.slippage_bps))
    results = [bt.run(prices, strat.signals(warm).loc[prices.index], name) for name, strat in BASELINES.items()]
    console.print(rich_table(results, title=f"Baselines {ticker} {start} → {end}"))


@app.command()
def memory(path: Path = typer.Option(None, help="memory JSONL (default from config)")) -> None:
    """Show decision-memory statistics."""
    from ailura.memory.store import DecisionMemory

    cfg = _config(None, None)
    mem = DecisionMemory(path or cfg.memory_path)
    console.print_json(json.dumps(mem.stats()))


if __name__ == "__main__":
    app()
