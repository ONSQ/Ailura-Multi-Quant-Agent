"""Ailura Multi-Quant Agent.

A provider-agnostic, offline-testable re-implementation of the TradingAgents
architecture (Xiao, Sun, Luo, Wang 2024; arXiv:2412.20138) with quantitative
improvements: point-in-time data guards, structured agent outputs, early-stopped
debates, volatility-targeted position sizing, outcome-based reflection memory,
and a transaction-cost-aware backtester with the paper's baselines.
"""

from ailura.config import AiluraConfig

__all__ = ["AiluraConfig", "__version__"]
__version__ = "0.1.0"
