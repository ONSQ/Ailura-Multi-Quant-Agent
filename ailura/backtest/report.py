"""Render backtest comparisons as a markdown table or a rich table."""

from __future__ import annotations

from collections.abc import Iterable

from ailura.backtest.engine import BacktestResult

COLUMNS = [
    ("name", "Strategy", "{}"),
    ("cumulative_return", "CR%", "{:+.2%}"),
    ("annualized_return", "AR%", "{:+.2%}"),
    ("sharpe", "Sharpe", "{:.2f}"),
    ("sortino", "Sortino", "{:.2f}"),
    ("max_drawdown", "MDD%", "{:.2%}"),
    ("calmar", "Calmar", "{:.2f}"),
    ("trades", "Trades", "{}"),
]


def results_table(results: Iterable[BacktestResult]) -> str:
    rows = [r.to_dict() for r in results]
    head = "| " + " | ".join(h for _, h, _ in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = ["| " + " | ".join(fmt.format(row.get(k, "")) for k, _, fmt in COLUMNS) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def rich_table(results: Iterable[BacktestResult], title: str = "Backtest"):
    from rich.table import Table

    t = Table(title=title)
    for _, h, _ in COLUMNS:
        t.add_column(h, justify="right" if h != "Strategy" else "left")
    for r in results:
        row = r.to_dict()
        t.add_row(*[fmt.format(row.get(k, "")) for k, _, fmt in COLUMNS])
    return t
