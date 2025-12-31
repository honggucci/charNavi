# 006_engine - Backtest Engine Module
# NOTE: backtest module is NOT imported here to avoid circular dependency
# Use: from wpcn._06_engine.backtest import DEFAULT_THETA, BacktestEngine
from .navigation import compute_navigation

__all__ = ['compute_navigation']
