"""
Rollback Side Effects 검증 테스트
=================================

rollback_to_champion() 실행 시 발생하는 모든 side effects를 검증합니다.

검증 대상:
1. ACTIVE가 Champion run_id로 복원되는지
2. Challenger가 히스토리에 retired로 기록되는지
3. challenger_run_id가 None으로 초기화되는지
4. Champion run_id는 유지되는지
5. 히스토리 파일이 올바르게 저장되는지

실행:
    pytest tests/test_rollback_side_effects.py -v
"""
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any


# ============================================================
# Mock 클래스 정의
# ============================================================

@dataclass
class MockParamConfidence:
    """Mock ParamConfidence"""
    oos_performance: Optional[float] = None
    confidence_score: float = 70.0
    oos_pending: bool = False

    def to_dict(self) -> Dict:
        return {
            "oos_performance": self.oos_performance,
            "confidence_score": self.confidence_score,
            "oos_pending": self.oos_pending,
        }


@dataclass
class MockParamPerformance:
    """Mock ParamPerformance"""
    oos: Optional[Dict] = None


@dataclass
class MockOptimizedParams:
    """Mock OptimizedParams"""
    params: Dict
    confidence: MockParamConfidence
    performance: MockParamPerformance
    run_id: str = ""


class MockParamStore:
    """Rollback side effects 추적용 Mock ParamStore"""

    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self._base_path = store_dir
        self._index: Dict[str, Dict] = {}
        self._active_params: Dict[str, MockOptimizedParams] = {}
        self._champion_params: Dict[str, MockOptimizedParams] = {}

        # Side effect 추적
        self.set_active_calls = []
        self.mark_as_champion_calls = []
        self.demote_champion_calls = []

    def get_active(self, symbol: str, timeframe: str) -> Optional[MockOptimizedParams]:
        key = f"{symbol}_{timeframe}"
        return self._active_params.get(key)

    def get_champion(self, symbol: str, timeframe: str) -> Optional[MockOptimizedParams]:
        key = f"{symbol}_{timeframe}"
        return self._champion_params.get(key)

    def set_active(self, symbol: str, timeframe: str, run_id: str) -> bool:
        """ACTIVE 설정 추적"""
        key = f"{symbol}_{timeframe}"
        self.set_active_calls.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat()
        })
        if key not in self._index:
            self._index[key] = {}
        self._index[key]["active"] = {"run_id": run_id}
        return True

    def mark_as_champion(self, symbol: str, timeframe: str, run_id: str) -> bool:
        """Champion 마킹 추적"""
        self.mark_as_champion_calls.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat()
        })
        key = f"{symbol}_{timeframe}"
        if key not in self._index:
            self._index[key] = {}
        self._index[key]["champion"] = {"run_id": run_id}
        return True

    def demote_champion(self, symbol: str, timeframe: str) -> bool:
        """Champion 해제 추적"""
        self.demote_champion_calls.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat()
        })
        key = f"{symbol}_{timeframe}"
        if key in self._index and "champion" in self._index[key]:
            del self._index[key]["champion"]
        return True

    def _set_active_params(self, symbol: str, timeframe: str, params: MockOptimizedParams):
        key = f"{symbol}_{timeframe}"
        self._active_params[key] = params

    def _set_champion_params(self, symbol: str, timeframe: str, params: MockOptimizedParams):
        key = f"{symbol}_{timeframe}"
        self._champion_params[key] = params


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def temp_store_dir():
    """임시 store 디렉토리"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_store(temp_store_dir):
    """Side effect 추적용 Mock ParamStore"""
    return MockParamStore(temp_store_dir)


@pytest.fixture
def version_manager(mock_store):
    """ParamVersionManager 인스턴스"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from wpcn._08_tuning.param_versioning import ParamVersionManager
    return ParamVersionManager(
        store=mock_store,
        symbol="BTCUSDT",
        market="futures",
        timeframe="15m"
    )


@pytest.fixture
def setup_champion_challenger(mock_store, version_manager):
    """Champion + Challenger 설정된 상태"""
    champion = MockOptimizedParams(
        params={"box_L": 96},
        confidence=MockParamConfidence(oos_performance=0.80, confidence_score=70.0),
        performance=MockParamPerformance(),
        run_id="20260101_120000"
    )
    challenger = MockOptimizedParams(
        params={"box_L": 100},
        confidence=MockParamConfidence(oos_performance=0.60, confidence_score=50.0),
        performance=MockParamPerformance(),
        run_id="20260108_120000"
    )

    mock_store._set_champion_params("BTCUSDT", "15m", champion)
    mock_store._set_active_params("BTCUSDT", "15m", challenger)
    version_manager._champion_run_id = "20260101_120000"
    version_manager._challenger_run_id = "20260108_120000"

    return version_manager, mock_store


# ============================================================
# Rollback Side Effects 테스트
# ============================================================

class TestRollbackSideEffects:
    """Rollback 실행 시 side effects 검증"""

    def test_active_restored_to_champion(self, setup_champion_challenger):
        """Side Effect 1: ACTIVE가 Champion run_id로 복원"""
        manager, store = setup_champion_challenger

        # 롤백 전 set_active 호출 횟수 기록
        calls_before = len(store.set_active_calls)

        manager.rollback_to_champion()

        # 롤백 후 set_active 호출됨
        assert len(store.set_active_calls) > calls_before

        # 마지막 set_active 호출이 Champion run_id로
        last_call = store.set_active_calls[-1]
        assert last_call["run_id"] == "20260101_120000"
        assert last_call["symbol"] == "BTCUSDT"
        assert last_call["timeframe"] == "15m"

    def test_challenger_recorded_as_retired(self, setup_champion_challenger):
        """Side Effect 2: Challenger가 히스토리에 retired로 기록"""
        manager, store = setup_champion_challenger

        manager.rollback_to_champion()

        history = manager.get_version_history()
        retired = [h for h in history if h["role"] == "retired" and h["run_id"] == "20260108_120000"]

        assert len(retired) == 1
        assert retired[0]["demoted_at"] is not None

    def test_challenger_run_id_becomes_none(self, setup_champion_challenger):
        """Side Effect 3: challenger_run_id가 None으로 초기화"""
        manager, store = setup_champion_challenger

        assert manager.challenger_run_id == "20260108_120000"

        manager.rollback_to_champion()

        assert manager.challenger_run_id is None

    def test_champion_run_id_preserved(self, setup_champion_challenger):
        """Side Effect 4: Champion run_id는 유지"""
        manager, store = setup_champion_challenger

        original_champion = manager.champion_run_id

        manager.rollback_to_champion()

        assert manager.champion_run_id == original_champion
        assert manager.champion_run_id == "20260101_120000"

    def test_history_file_saved(self, setup_champion_challenger, temp_store_dir):
        """Side Effect 5: 히스토리 파일이 저장됨"""
        manager, store = setup_champion_challenger

        manager.rollback_to_champion()

        history_file = temp_store_dir / "BTCUSDT_15m_version_history.json"
        assert history_file.exists()

        with open(history_file, "r") as f:
            data = json.load(f)

        assert data["champion_run_id"] == "20260101_120000"
        assert data["challenger_run_id"] is None


# ============================================================
# Promote Side Effects 테스트
# ============================================================

class TestPromoteSideEffects:
    """Promote 실행 시 side effects 검증"""

    def test_mark_as_champion_called(self, mock_store, version_manager):
        """Side Effect: mark_as_champion이 호출됨"""
        challenger = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=0.85, confidence_score=75.0),
            performance=MockParamPerformance(),
            run_id="20260108_120000"
        )
        mock_store._set_active_params("BTCUSDT", "15m", challenger)
        version_manager._challenger_run_id = "20260108_120000"

        version_manager.promote_challenger()

        assert len(mock_store.mark_as_champion_calls) >= 1
        last_call = mock_store.mark_as_champion_calls[-1]
        assert last_call["run_id"] == "20260108_120000"

    def test_old_champion_moved_to_history(self, setup_champion_challenger):
        """Side Effect: 기존 Champion이 히스토리로 이동"""
        manager, store = setup_champion_challenger

        # Challenger OOS를 승격 조건으로 수정
        challenger = store._active_params["BTCUSDT_15m"]
        challenger.confidence.oos_performance = 0.82

        manager.promote_challenger()

        history = manager.get_version_history()
        retired = [h for h in history if h["role"] == "retired" and h["run_id"] == "20260101_120000"]

        assert len(retired) == 1

    def test_challenger_becomes_champion(self, setup_champion_challenger):
        """Side Effect: Challenger가 Champion으로 승격"""
        manager, store = setup_champion_challenger

        # Challenger OOS를 승격 조건으로 수정
        challenger = store._active_params["BTCUSDT_15m"]
        challenger.confidence.oos_performance = 0.82

        manager.promote_challenger()

        assert manager.champion_run_id == "20260108_120000"
        assert manager.challenger_run_id is None


# ============================================================
# 연속 작업 Side Effects 테스트
# ============================================================

class TestSequentialOperations:
    """연속 작업 시 side effects 일관성 검증"""

    def test_multiple_challengers_then_rollback(self, mock_store, version_manager):
        """여러 Challenger 등록 후 롤백"""
        # Champion 설정
        champion = MockOptimizedParams(
            params={"box_L": 96},
            confidence=MockParamConfidence(oos_performance=0.80, confidence_score=70.0),
            performance=MockParamPerformance(),
            run_id="20260101_120000"
        )
        mock_store._set_champion_params("BTCUSDT", "15m", champion)
        version_manager._champion_run_id = "20260101_120000"

        # Challenger 1 등록
        version_manager.register_challenger("20260108_120000")

        # Challenger 2 등록 (1을 교체)
        version_manager.register_challenger("20260115_120000")

        # 롤백
        challenger = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=0.60, confidence_score=50.0),
            performance=MockParamPerformance(),
            run_id="20260115_120000"
        )
        mock_store._set_active_params("BTCUSDT", "15m", challenger)

        version_manager.rollback_to_champion()

        # 검증
        assert version_manager.challenger_run_id is None
        assert version_manager.champion_run_id == "20260101_120000"

        # 히스토리에 retired가 있음 (rollback에서 retired로 기록)
        history = version_manager.get_version_history()
        retired = [h for h in history if h["role"] == "retired"]
        # 실제로는 register_challenger에서 old가 retired로, rollback에서도 retired로
        assert len(retired) >= 1

    def test_promote_then_rollback(self, mock_store, version_manager):
        """승격 후 다음 Challenger 롤백"""
        # 초기 Champion 설정
        champion1 = MockOptimizedParams(
            params={"box_L": 96},
            confidence=MockParamConfidence(oos_performance=0.75, confidence_score=65.0),
            performance=MockParamPerformance(),
            run_id="20260101_120000"
        )
        mock_store._set_champion_params("BTCUSDT", "15m", champion1)
        version_manager._champion_run_id = "20260101_120000"

        # Challenger 1 등록 및 승격
        challenger1 = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=0.80, confidence_score=70.0),
            performance=MockParamPerformance(),
            run_id="20260108_120000"
        )
        mock_store._set_active_params("BTCUSDT", "15m", challenger1)
        version_manager._challenger_run_id = "20260108_120000"

        version_manager.promote_challenger()

        # 이제 Champion = 20260108_120000
        assert version_manager.champion_run_id == "20260108_120000"
        mock_store._set_champion_params("BTCUSDT", "15m", challenger1)

        # Challenger 2 등록 및 롤백
        challenger2 = MockOptimizedParams(
            params={"box_L": 110},
            confidence=MockParamConfidence(oos_performance=0.50, confidence_score=40.0),
            performance=MockParamPerformance(),
            run_id="20260115_120000"
        )
        mock_store._set_active_params("BTCUSDT", "15m", challenger2)
        version_manager._challenger_run_id = "20260115_120000"

        version_manager.rollback_to_champion()

        # 검증: Champion은 승격된 20260108_120000 유지
        assert version_manager.champion_run_id == "20260108_120000"
        assert version_manager.challenger_run_id is None


# ============================================================
# Edge Case 테스트
# ============================================================

class TestEdgeCases:
    """Edge case에서의 side effects"""

    def test_rollback_without_champion(self, mock_store, version_manager):
        """Champion 없이 롤백 시도"""
        # Challenger만 있고 Champion 없음
        challenger = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=0.60, confidence_score=50.0),
            performance=MockParamPerformance(),
            run_id="20260108_120000"
        )
        mock_store._set_active_params("BTCUSDT", "15m", challenger)
        version_manager._challenger_run_id = "20260108_120000"
        version_manager._champion_run_id = None

        result = version_manager.rollback_to_champion()

        # Challenger가 retired로 기록
        history = version_manager.get_version_history()
        retired = [h for h in history if h["role"] == "retired"]
        assert len(retired) >= 1

        # set_active는 호출되지 않음 (Champion 없으므로)
        # 또는 호출되어도 run_id=None
        assert version_manager.challenger_run_id is None

    def test_rollback_without_challenger(self, mock_store, version_manager):
        """Challenger 없이 롤백 시도"""
        champion = MockOptimizedParams(
            params={"box_L": 96},
            confidence=MockParamConfidence(oos_performance=0.80, confidence_score=70.0),
            performance=MockParamPerformance(),
            run_id="20260101_120000"
        )
        mock_store._set_champion_params("BTCUSDT", "15m", champion)
        version_manager._champion_run_id = "20260101_120000"
        version_manager._challenger_run_id = None

        result = version_manager.rollback_to_champion()

        assert result is False  # 롤백할 것이 없음

    def test_oos_performance_preserved_in_history(self, setup_champion_challenger):
        """Side Effect: oos_performance가 히스토리에 기록됨"""
        manager, store = setup_champion_challenger

        manager.rollback_to_champion()

        history = manager.get_version_history()
        retired = [h for h in history if h["run_id"] == "20260108_120000"][0]

        # Challenger의 oos_performance가 기록됨
        assert retired.get("oos_performance") == 0.60

    def test_confidence_score_preserved_in_history(self, setup_champion_challenger):
        """Side Effect: confidence_score가 히스토리에 기록됨"""
        manager, store = setup_champion_challenger

        manager.rollback_to_champion()

        history = manager.get_version_history()
        retired = [h for h in history if h["run_id"] == "20260108_120000"][0]

        assert retired.get("confidence_score") == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
