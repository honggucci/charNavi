# 006_engine - Backtest Engine Module
# NOTE: backtest module is NOT imported here to avoid circular dependency
# Use:
#   from wpcn._06_engine.backtest import DEFAULT_THETA, BacktestEngine, load_theta_from_store
#   from wpcn._06_engine.backtest import load_champion_theta  # v2.2 Champion 우선 로드
#   from wpcn._06_engine.backtest import load_theta_with_run_id  # v2.2.1 run_id 추적
#   from wpcn._06_engine.backtest import normalize_symbol  # v2.2.1 심볼 정규화
from .navigation import compute_navigation

__all__ = ['compute_navigation']
