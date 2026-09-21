# TradingAgents: research summary, gap analysis, and implementation plan

## 1. What the paper proposes

**Paper:** *TradingAgents: Multi-Agents LLM Financial Trading Framework*,
Yijia Xiao, Edward Sun, Di Luo, Wei Wang (UCLA, MIT, Tauric Research),
arXiv:2412.20138 (Dec 2024; v7 June 2025).
**Code:** github.com/TauricResearch/TradingAgents (Apache-2.0, Python, LangGraph).

The framework simulates a trading firm as a graph of LLM agents with
specialised roles, arguing that role separation plus structured reports plus
adversarial debate beats a single-prompt agent.

| Team | Agents | Input | Output |
|---|---|---|---|
| Analysts | Fundamentals, Sentiment (social), News (macro + company), Technical | tool calls (prices, indicators, news, filings, social feeds) | one report each |
| Researchers | Bullish, Bearish | analyst reports | multi-round debate, judged by a Research Manager |
| Trader | Trader | manager's plan + reports | concrete BUY/SELL/HOLD proposal |
| Risk management | Aggressive, Conservative, Neutral | trader proposal + reports | three-way debate |
| Fund manager | Portfolio Manager | risk debate | final approved decision |

**Communication protocol.** Analysts hand over *structured reports* (to avoid
long-context "telephone game" degradation); researchers and risk analysts use
*natural-language debate* for a bounded number of rounds. A quick-thinking
model does the high-volume work; a deep-thinking model judges debates and
makes the final call.

**Memory / reflection.** After outcomes are known, agents reflect on past
decisions and store lessons that are retrieved for similar future situations.

**Experiment.** AAPL, GOOGL, AMZN; test window 1 Jan – 29 Mar 2024; quick
model `gpt-4o-mini`, deep model `o1-preview`; baselines Buy & Hold, MACD,
KDJ&RSI, ZMR (z-score mean reversion), SMA. Reported (cumulative return /
Sharpe): AAPL 26.62 % / 8.21, GOOGL 24.36 % / 6.39, AMZN 23.21 % / 5.60; every
baseline had Sharpe below 3 and larger drawdowns.

**Limitations the authors and reviewers flag.** Three tickers over three
months; costly (dozens of LLM calls per decision); never deployed live; a
Sharpe of 6–8 on single stocks is far outside what survives realistic
execution, so the evaluation needs replication with strict point-in-time data
and transaction costs.

## 2. What the reference implementation does today (v0.5)

* LangGraph state machine; analysts run agentic tool loops (`get_stock_data`,
  indicators, news, insider data, balance sheet, ...); `Msg Clear` nodes
  between analysts.
* `DEFAULT_CONFIG` with `deep_think_llm`, `quick_think_llm`,
  `max_debate_rounds`, `max_risk_discuss_rounds`, vendor chains with
  fall-through (`yfinance,alpha_vantage`), many LLM providers.
* Five-tier rating (Buy / Overweight / Hold / Underweight / Sell) parsed
  deterministically from the Portfolio Manager's markdown; unparseable ⇒
  `REVIEW`, never a silent Hold.
* Decision log at `~/.tradingagents/memory/trading_memory.md`; on later runs
  realised returns are fetched, a short reflection is written, and lessons are
  injected into the Portfolio Manager's context (point-in-time filtered).
* `run_backtest()` scores decisions by hit rate and mean alpha over a holding
  window against a regional benchmark. It does **not** simulate a portfolio,
  costs, or produce the paper's CR/AR/Sharpe/MDD table.

## 3. Gap analysis (what stops this being a research-grade system)

| # | Gap | Consequence |
|---|---|---|
| G1 | No hard look-ahead guard: analysts call live tools; Yahoo fundamentals/news are current, not as-of | inflated backtests |
| G2 | Free-text hand-offs between most agents | brittle parsing, silent drift, hard to evaluate components |
| G3 | Debate always runs the configured rounds | cost with no information gain |
| G4 | Binary action; sizing left to prose | can't express conviction, no risk budget |
| G5 | No portfolio backtester with costs; no reproduction of the paper's metrics table | claims can't be verified |
| G6 | Every run requires network + API keys | no CI, no deterministic tests |
| G7 | Reflection lesson is prose, retrieval is by ticker only | lessons don't transfer across regimes |
| G8 | Heavy dependency surface (LangChain, LangGraph, many vendor SDKs) | hard to audit, slow to change |

## 4. What this repository implements (v0.1, done)

| Improvement | Where | Closes |
|---|---|---|
| `PointInTimeProvider`: every input clipped to the decision date; requests past it raise `LookaheadError`; tests prove decisions are identical with future data removed | `ailura/data/base.py`, `tests/test_graph.py::test_no_lookahead_in_decision` | G1 |
| Inputs are pre-fetched once and handed to analysts (no per-analyst tool loops): 14 LLM calls per decision at default settings instead of ~25 | `ailura/graph/orchestrator.py` | G3, G8 |
| Pydantic schema for every role; native structured outputs on Anthropic, JSON-mode + local validation elsewhere; loud `StructuredOutputError` | `ailura/agents/schemas.py`, `ailura/llm/*` | G2 |
| Debate early-stop: ends when neither side adds evidence or the round's evidence is mostly a repeat (token-overlap novelty) | `TradingGraph._debate` | G3 |
| Signed `conviction` ∈ [−1, 1] and `position_fraction` throughout; trader sizes with vol targeting (target vol / realised vol, capped); risk personas argue over size and stop; PM output is post-checked (`sanitize_decision`) | agents, `graph/orchestrator.py` | G4 |
| Daily backtester with commission + slippage, rebalance threshold, next-close option; CR / AR / Sharpe / Sortino / MDD / Calmar / hit-rate; the paper's B&H, MACD, KDJ&RSI, ZMR, SMA baselines; `AgentStrategy` wrapper that runs the graph day-by-day under the PIT guard | `ailura/backtest/` | G5 |
| `HeuristicLLM`: deterministic role-dispatched stand-in that produces schema-valid outputs from the structured context; the whole pipeline, memory loop and CLI run in CI with no network | `ailura/llm/heuristic.py`, `tests/` | G6 |
| Outcome memory: JSONL log, settled *as of* each decision date (online learning in backtests without leakage), lessons tagged by trend/vol regime and retrieved by ticker + tag overlap; `Reflector` produces a structured lesson | `ailura/memory/` | G7 |
| Minimal dependencies (pydantic, pandas, numpy, typer, rich); provider SDKs optional; cost ledger per role/model | `pyproject.toml`, `ailura/llm/base.py` | G8 |

The `HeuristicLLM` also serves as a **rules-ensemble baseline** for the same
pipeline, so the marginal value of an LLM can be measured rather than assumed.

## 5. Roadmap

### Phase 1 — Replicate the paper honestly (next)
1. Load AAPL / GOOGL / AMZN daily bars for 2023-01 → 2024-03 via the CSV
   provider (or yfinance where allowed) and run `ailura backtest` for
   1 Jan – 29 Mar 2024 with `claude-opus-5` (deep) / `claude-sonnet-5` (quick).
2. Report the paper's table (CR %, AR %, Sharpe, MDD %) for the five baselines,
   the heuristic pipeline, and the LLM pipeline under binary and conviction
   sizing, with 3 bps round-trip costs and next-close execution.
3. Build historical news/social/fundamentals feeds as point-in-time JSONL
   (e.g. from an archived news dataset and SEC filing dates) so the text
   analysts are evaluated without leakage.
4. Add a `paper-replication` CLI command that runs the grid and writes
   `results/replication.md`.

### Phase 2 — Make the evaluation statistically meaningful
* Widen to 20–30 liquid tickers and 2+ years; walk-forward windows.
* Bootstrap confidence intervals on Sharpe; deflated Sharpe for multiple
  comparisons; report turnover and cost sensitivity.
* Ablations: no debate, no risk team, no memory, single-model, analyst subsets
  (`AILURA_ANALYSTS`) — the pipeline already supports all of these via config.
* Calibration: does `conviction` predict realised alpha? (reliability plot.)

### Phase 3 — Portfolio, not single names
* Multi-asset `PortfolioContext`: gross/net limits, sector caps, correlation-
  aware sizing; a cross-sectional Portfolio Manager ranking candidates.
* Regime-conditioned prompts and memory (already tagged by trend/vol regime).
* Execution model: VWAP participation, borrow cost for shorts.

### Phase 4 — Cost, latency, operations
* Batch API for backtests (50 % cost), prompt-cache-friendly prompt layout
  (static system prompts first — already done), per-role effort tuning.
* Async fan-out of the four analysts; checkpoint/resume for long grids.
* Paper-trading loop with a broker adapter and a daily routine; kill-switches
  keyed on `REVIEW` signals and data-quality scores.

## 6. Evaluation protocol (how we will judge any change)

1. Same tickers, dates, costs and execution model for every strategy.
2. Strict PIT: a decision is invalid if it changes when future data is removed
   (automated in tests).
3. Headline metric: annualised Sharpe net of costs with bootstrap CI; secondary:
   MDD, turnover, hit rate by rating, calibration of conviction.
4. Cost per decision and wall-clock reported next to every result.
5. Beat the `HeuristicLLM` pipeline and Buy & Hold before claiming an edge.

## 7. Known limitations of v0.1

* Real-provider clients are covered by fake-SDK tests only; not yet run live
  from this repo.
* Yahoo news/fundamentals are present-day snapshots (documented); use CSV
  feeds for historical text.
* Synthetic data is for plumbing and tests, not for drawing conclusions.
