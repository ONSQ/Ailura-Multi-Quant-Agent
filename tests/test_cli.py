from typer.testing import CliRunner

from ailura.cli import app

runner = CliRunner()


def test_cli_analyze_and_baselines(tmp_path, monkeypatch):
    monkeypatch.setenv("AILURA_RESULTS_DIR", str(tmp_path / "r"))
    monkeypatch.setenv("AILURA_MEMORY_PATH", str(tmp_path / "m.jsonl"))
    res = runner.invoke(app, ["analyze", "AAPL", "--date", "2024-03-01"])
    assert res.exit_code == 0, res.output
    assert "FINAL DECISION" in res.output
    res = runner.invoke(app, ["baselines", "AAPL", "--start", "2024-01-01", "--end", "2024-02-15"])
    assert res.exit_code == 0, res.output
    assert "KDJ&RSI" in res.output
    res = runner.invoke(app, ["memory"])
    assert res.exit_code == 0 and '"total": 1' in res.output


def test_cli_backtest_small(tmp_path, monkeypatch):
    monkeypatch.setenv("AILURA_RESULTS_DIR", str(tmp_path / "r"))
    out = tmp_path / "bt.json"
    res = runner.invoke(app, ["backtest", "MSFT", "--start", "2024-02-01", "--end", "2024-02-20", "--rebalance-every", "5", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert out.exists() and "Ailura" in res.output
