"""
TradingAgents Evaluation Framework

This module provides comprehensive evaluation and backtesting capabilities
for the TradingAgents multi-agent trading system.
"""

from .backtesting import BacktestEngine, BacktestConfig
from .portfolio import Portfolio, PortfolioManager
from .metrics import PerformanceMetrics, MetricsCalculator
from .evaluator import TradingEvaluator
from .data_manager import DataManager
from .visualization import PerformanceVisualizer

__all__ = [
    "BacktestEngine",
    "BacktestConfig", 
    "Portfolio",
    "PortfolioManager",
    "PerformanceMetrics",
    "MetricsCalculator",
    "TradingEvaluator",
    "DataManager",
    "PerformanceVisualizer"
]

__version__ = "1.0.0"
