"""
_04_execution: 실행 및 시뮬레이션 모듈
======================================

백테스트/라이브 실행 관련 모듈.

v3 추가:
- state_persistence: Crash Recovery용 상태 영속화
"""

# State Persistence (v3: Crash Recovery)
from .state_persistence import (
    StateManager,
    PersistedPosition,
    PersistedTrade,
    PersistedState,
    get_state_manager,
    recover_from_crash,
)

__all__ = [
    # State Persistence
    "StateManager",
    "PersistedPosition",
    "PersistedTrade",
    "PersistedState",
    "get_state_manager",
    "recover_from_crash",
]
