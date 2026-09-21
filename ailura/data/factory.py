from __future__ import annotations

from ailura.config import AiluraConfig
from ailura.data.base import MarketDataProvider


def build_provider(config: AiluraConfig) -> MarketDataProvider:
    if config.data_provider == "synthetic":
        from ailura.data.synthetic import SyntheticProvider

        return SyntheticProvider(seed=config.synthetic_seed)
    if config.data_provider == "csv":
        from ailura.data.csv_provider import CSVProvider

        return CSVProvider(config.data_dir)
    if config.data_provider == "yfinance":
        from ailura.data.yfinance_provider import YFinanceProvider

        return YFinanceProvider(config.data_dir)
    raise ValueError(f"unknown data_provider {config.data_provider!r}")
