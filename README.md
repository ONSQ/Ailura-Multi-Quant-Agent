# Ailura Multi-Quant Agent

A provider-agnostic, **offline-testable** re-implementation of the
[TradingAgents](https://arxiv.org/abs/2412.20138) multi-agent LLM trading
framework (Xiao, Sun, Luo, Wang; UCLA / MIT / Tauric Research), extended with
the quantitative controls a research desk actually needs: point-in-time data
guards, structured agent outputs, early-stopped debates, volatility-targeted
sizing, outcome-based reflection memory, and a transaction-cost-aware
backtester that ships with the paper's five baselines.

See [`docs/PLAN.md`](docs/PLAN.md) for the research summary, gap analysis and
roadmap.

## Pipeline

```
 point-in-time data ─┬─ Market analyst ──────┐
 (prices, indicators │─ News analyst ────────┤   Bull ⇄ Bear debate      Research      Trader
  news, social,      │─ Sentiment analyst ───┼─▶ (early-stops when   ─▶  Manager  ─▶ (entry/stop/
  fundamentals)      └─ Fundamentals analyst ┘    nothing new is said)   (5-tier)      size)
                                                                                          │
                              Decision memory ◀── Reflector ◀── realized alpha            ▼
                              (lessons resolved                         Aggressive / Conservative / Neutral
                               strictly before                                   risk debate
                               the decision date)                                     │
                                                                                      ▼
                                                                    Portfolio Manager ─▶ FINAL DECISION
                                                                    (rating, action, size, stops, triggers)
```

Every hand-off is a validated pydantic object (`ailura/agents/schemas.py`),
and the final decision is post-checked for invariants (action ↔ rating,
position limits, stop on the right side of price).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core: pydantic, pandas, numpy, typer, rich
pytest                             # 41 offline, deterministic tests

# One decision, fully offline (synthetic data + deterministic heuristic "LLM")
ailura analyze AAPL --date 2024-03-01

# Agent pipeline vs the paper's baselines over a window
ailura backtest AAPL --start 2024-01-01 --end 2024-03-29 --rebalance-every 1
ailura baselines AAPL --start 2024-01-01 --end 2024-03-29
```

### Using a real model

```bash
pip install -e ".[anthropic]"      # or .[openai] for any OpenAI-compatible endpoint
export ANTHROPIC_API_KEY=...       # or run `ant auth login`
export AILURA_LLM_PROVIDER=anthropic
export AILURA_DEEP_MODEL=claude-opus-5      # research manager + portfolio manager
export AILURA_QUICK_MODEL=claude-sonnet-5   # analysts, researchers, trader, risk team
ailura analyze NVDA --date 2026-09-18 --provider yfinance
```

The Anthropic client uses adaptive thinking, an `effort` level per tier,
prompt caching on the system prompt, native structured outputs, and
server-side refusal fallbacks (`AILURA_ANTHROPIC_FALLBACKS=false` to disable).

### Using real market data

* `--provider yfinance` (`pip install -e ".[yfinance]"`): prices are cached to
  `data/cache/`. Yahoo's news/fundamentals are **not** point-in-time; treat
  those analysts' output as indicative for historical dates.
* `--provider csv`: drop `data/<TICKER>.csv` (Date,Open,High,Low,Close,Volume)
  plus optional `<TICKER>_news.jsonl`, `<TICKER>_social.jsonl`,
  `<TICKER>_fundamentals.json` for honest historical text feeds. The format is
  documented in `ailura/data/csv_provider.py`.

## Configuration

All fields of `AiluraConfig` (`ailura/config.py`) can be set with
`AILURA_<FIELD>` environment variables; see `.env.example`. Key knobs:

| Field | Default | Meaning |
|---|---|---|
| `max_debate_rounds` / `early_stop_debate` | 2 / true | bull-bear rounds; stop when neither side adds evidence |
| `max_risk_rounds` | 1 | aggressive/conservative/neutral rounds |
| `holding_days` | 5 | window used to settle a decision and write a lesson |
| `target_annual_vol` / `max_position_fraction` / `allow_short` | 0.20 / 1.0 / false | sizing limits handed to the trader and risk team |
| `min_hold_days` | 3 | an open position isn't cut/reversed this soon unless \|conviction\| ≥ 0.6 |
| `trend_guard` / `downtrend_max_fraction` | true / 0.5 | cap long size while SMA50 < SMA200 (caps size, never creates a signal) |
| `commission_bps` / `slippage_bps` | 1 / 2 | backtest costs |
| `llm_knowledge_cutoff` | unset | model training cutoff; real-LLM backtests starting before it get a warning |
| `memory_path` | `.ailura/memory.jsonl` | decision log; `None` for in-memory |

## Getting calls right: rules this repo enforces

These come from replication studies of LLM trading agents (FINSABER
arXiv:2505.07078, *Profit Mirage* arXiv:2510.07920, AlphaForgeBench
arXiv:2602.18481) and from upstream TradingAgents issues (#203, #1170, #1466):

* **HOLD keeps the position.** It is never an exit; close a long with
  Underweight/Sell. (Treating HOLD as "go flat" made the book flip long↔cash
  on tiny conviction changes.)
* **Hysteresis outside the LLM.** LLM agents flip between adjacent decisions
  even at temperature 0. A minimum holding period, a 5% rebalance band and a
  trend guard are enforced in code (`apply_position_rules`); overrides are
  tagged in the executive summary and counted in the backtest's `decisions:`
  line, next to the BUY/HOLD/SELL base rates and position flips.
* **Realistic fills.** `ailura backtest` executes on the next close by
  default (`--execution close` for the paper's optimistic convention).
* **No future fundamentals.** Yahoo fundamentals are present-day only and are
  withheld for historical dates; live runs refresh a stale cache and ignore a
  partial intraday bar.
* **Inaction is reviewed.** The reflector grades a flat Hold that missed a >3%
  move as incorrect, so memory doesn't teach the pipeline to sit in cash.
* **Memorization.** A real LLM has seen the prices of any window before its
  training cutoff, so backtest only after it (set `AILURA_LLM_KNOWLEDGE_CUTOFF`).
  Judge any edge against Buy & Hold and the heuristic pipeline, net of costs,
  over many tickers. Three months on three names proves nothing.

## Layout

```
ailura/
  config.py            AiluraConfig (+ env overrides)
  llm/                 LLMClient protocol, HeuristicLLM (offline), Anthropic, OpenAI-compatible, cost ledger
  data/                provider protocol, PointInTimeProvider guard, synthetic / CSV / yfinance, indicators
  agents/              schemas + analysts, researchers, trader, risk team, portfolio manager
  graph/               TradingState, TradingGraph orchestrator, rating parser
  memory/              DecisionMemory (JSONL, PIT recall) + Reflector
  backtest/            engine, metrics, paper baselines, AgentStrategy wrapper, reports
  cli.py               ailura analyze | backtest | baselines | memory
tests/                 offline, deterministic; CI runs them on 3.10 and 3.12
docs/PLAN.md           research summary, gap analysis, roadmap
```

## Status

v0.1: architecture complete and tested offline; real-provider clients are
exercised against fake SDK objects in tests and have **not** yet been run
against live APIs from this repository. Nothing here is investment advice.
