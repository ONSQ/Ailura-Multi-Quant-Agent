from __future__ import annotations

from datetime import date

import pytest

from ailura.config import AiluraConfig
from ailura.data.synthetic import SyntheticProvider
from ailura.graph.orchestrator import TradingGraph
from ailura.memory.store import DecisionMemory


@pytest.fixture
def provider() -> SyntheticProvider:
    return SyntheticProvider(seed=7)


@pytest.fixture
def config(tmp_path) -> AiluraConfig:
    return AiluraConfig(results_dir=tmp_path / "results", memory_path=None)


@pytest.fixture
def graph(config, provider) -> TradingGraph:
    return TradingGraph(config, provider=provider, memory=DecisionMemory(None))


@pytest.fixture
def a_date() -> date:
    return date(2024, 3, 1)
