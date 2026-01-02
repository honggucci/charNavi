"""
param_versioning.py Unit Tests (v2.2.1)
=======================================

테스트 대상:
- ParamVersionManager
- Champion/Challenger 승격/롤백 로직
- 헬퍼 함수 (_get_oos_performance, _get_confidence_score, _is_oos_pending)

실행:
    pytest tests/test_param_versioning.py -v
"""
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Optional, Dict, Any


# ============================================================
# Mock 클래스 정의 (실제 ParamStore 의존성 제거)
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
    """Mock ParamStore for testing"""

    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self._base_path = store_dir
        self._index: Dict[str, Dict] = {}
        self._active_params: Dict[str, MockOptimizedParams] = {}
        self._champion_params: Dict[str, MockOptimizedParams] = {}

    def get_active(self, symbol: str, timeframe: str) -> Optional[MockOptimizedParams]:
        key = f"{symbol}_{timeframe}"
        return self._active_params.get(key)

    def get_champion(self, symbol: str, timeframe: str) -> Optional[MockOptimizedParams]:
        key = f"{symbol}_{timeframe}"
        return self._champion_params.get(key)

    def set_active(self, symbol: str, timeframe: str, run_id: str) -> bool:
        key = f"{symbol}_{timeframe}"
        # 실제로는 파일에서 로드하지만, mock에서는 성공으로 처리
        if key not in self._index:
            self._index[key] = {}
        self._index[key]["active"] = {"run_id": run_id}
        return True

    def mark_as_champion(self, symbol: str, timeframe: str, run_id: str) -> bool:
        key = f"{symbol}_{timeframe}"
        if key not in self._index:
            self._index[key] = {}
        self._index[key]["champion"] = {"run_id": run_id}
        return True

    def demote_champion(self, symbol: str, timeframe: str) -> bool:
        key = f"{symbol}_{timeframe}"
        if key in self._index and "champion" in self._index[key]:
            del self._index[key]["champion"]
        return True

    # 테스트용 헬퍼
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
    """임시 store 디렉토리 생성"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_store(temp_store_dir):
    """Mock ParamStore 인스턴스"""
    return MockParamStore(temp_store_dir)


@pytest.fixture
def version_manager(mock_store):
    """ParamVersionManager 인스턴스"""
    # 실제 모듈 import
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from wpcn._08_tuning.param_versioning import ParamVersionManager
    return ParamVersionManager(
        store=mock_store,
        symbol="BTCUSDT",
        market="futures",
        timeframe="15m"
    )


# ============================================================
# 헬퍼 함수 테스트
# ============================================================

class TestHelperFunctions:
    """헬퍼 함수 테스트"""

    def test_get_oos_performance_with_valid_confidence(self):
        """유효한 confidence에서 oos_performance 추출"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from wpcn._08_tuning.param_versioning import _get_oos_performance

        param = MockOptimizedParams(
            params={},
            confidence=MockParamConfidence(oos_performance=0.85),
            performance=MockParamPerformance()
        )

        result = _get_oos_performance(param)
        assert result == 0.85

    def test_get_oos_performance_with_none_param(self):
        """None 파라미터 처리"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from wpcn._08_tuning.param_versioning import _get_oos_performance

        result = _get_oos_performance(None)
        assert result is None

    def test_get_oos_performance_with_none_confidence(self):
        """confidence가 None인 경우"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from wpcn._08_tuning.param_versioning import _get_oos_performance

        param = Mock()
        param.confidence = None

        result = _get_oos_performance(param)
        assert result is None

    def test_get_confidence_score_with_valid_value(self):
        """유효한 confidence_score 추출"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from wpcn._08_tuning.param_versioning import _get_confidence_score

        param = MockOptimizedParams(
            params={},
            confidence=MockParamConfidence(confidence_score=75.5),
            performance=MockParamPerformance()
        )

        result = _get_confidence_score(param)
        assert result == 75.5

    def test_is_oos_pending_true(self):
        """oos_pending=True 체크"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from wpcn._08_tuning.param_versioning import _is_oos_pending

        param = MockOptimizedParams(
            params={},
            confidence=MockParamConfidence(oos_pending=True),
            performance=MockParamPerformance()
        )

        result = _is_oos_pending(param)
        assert result is True

    def test_is_oos_pending_false(self):
        """oos_pending=False 체크"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from wpcn._08_tuning.param_versioning import _is_oos_pending

        param = MockOptimizedParams(
            params={},
            confidence=MockParamConfidence(oos_pending=False),
            performance=MockParamPerformance()
        )

        result = _is_oos_pending(param)
        assert result is False

    def test_is_oos_pending_none_param(self):
        """None 파라미터는 pending으로 처리"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from wpcn._08_tuning.param_versioning import _is_oos_pending

        result = _is_oos_pending(None)
        assert result is True


# ============================================================
# ComparisonResult 테스트
# ============================================================

class TestComparisonResult:
    """ComparisonResult 데이터클래스 테스트"""

    def test_comparison_result_to_dict(self):
        """to_dict 직렬화 테스트"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from wpcn._08_tuning.param_versioning import ComparisonResult

        result = ComparisonResult(
            champion_run_id="20260101_120000",
            challenger_run_id="20260108_120000",
            champion_oos=0.80,
            challenger_oos=0.85,
            champion_confidence=70.0,
            challenger_confidence=75.0,
            oos_improvement=0.0625,
            confidence_improvement=0.0714,
            should_promote=True,
            reason="Test"
        )

        d = result.to_dict()
        assert d["champion_run_id"] == "20260101_120000"
        assert d["challenger_oos"] == 0.85
        assert d["should_promote"] is True


# ============================================================
# ParamVersionManager 테스트
# ============================================================

class TestParamVersionManager:
    """ParamVersionManager 핵심 로직 테스트"""

    def test_register_challenger_success(self, version_manager, mock_store):
        """Challenger 등록 성공"""
        result = version_manager.register_challenger("20260102_120000")

        assert result is True
        assert version_manager.challenger_run_id == "20260102_120000"

    def test_register_challenger_replaces_old(self, version_manager, mock_store):
        """기존 Challenger를 새 것으로 교체"""
        # 기존 challenger 설정
        old_params = MockOptimizedParams(
            params={"box_L": 96},
            confidence=MockParamConfidence(oos_performance=0.75),
            performance=MockParamPerformance(),
            run_id="20260101_120000"
        )
        mock_store._set_active_params("BTCUSDT", "15m", old_params)
        version_manager._challenger_run_id = "20260101_120000"

        # 새 challenger 등록
        result = version_manager.register_challenger("20260102_120000")

        assert result is True
        assert version_manager.challenger_run_id == "20260102_120000"
        # 히스토리에 old가 retired로 기록되었는지
        history = version_manager.get_version_history()
        retired = [h for h in history if h["role"] == "retired"]
        assert len(retired) >= 1

    def test_compare_performance_no_challenger(self, version_manager):
        """Challenger 없을 때 비교 결과"""
        result = version_manager.compare_performance()

        assert result.should_promote is False
        assert "No challenger" in result.reason

    def test_compare_performance_first_version_auto_promote(self, version_manager, mock_store):
        """첫 번째 버전은 자동 승격"""
        # Challenger만 있고 Champion 없음
        challenger = MockOptimizedParams(
            params={"box_L": 96},
            confidence=MockParamConfidence(oos_performance=0.80),
            performance=MockParamPerformance(),
            run_id="20260102_120000"
        )
        mock_store._set_active_params("BTCUSDT", "15m", challenger)
        version_manager._challenger_run_id = "20260102_120000"

        result = version_manager.compare_performance()

        assert result.should_promote is True
        assert "First version" in result.reason

    def test_compare_performance_promote_condition(self, version_manager, mock_store):
        """승격 조건 충족 (OOS >= 0.95, Conf >= 0.90)"""
        champion = MockOptimizedParams(
            params={"box_L": 96},
            confidence=MockParamConfidence(oos_performance=0.80, confidence_score=70.0),
            performance=MockParamPerformance(),
            run_id="20260101_120000"
        )
        challenger = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=0.82, confidence_score=68.0),
            performance=MockParamPerformance(),
            run_id="20260108_120000"
        )

        mock_store._set_champion_params("BTCUSDT", "15m", champion)
        mock_store._set_active_params("BTCUSDT", "15m", challenger)
        version_manager._champion_run_id = "20260101_120000"
        version_manager._challenger_run_id = "20260108_120000"

        result = version_manager.compare_performance()

        # 0.82/0.80 = 1.025 >= 0.95 ✓
        # 68/70 = 0.97 >= 0.90 ✓
        assert result.should_promote is True

    def test_compare_performance_rollback_condition(self, version_manager, mock_store):
        """롤백 조건 (OOS < 0.80)"""
        champion = MockOptimizedParams(
            params={"box_L": 96},
            confidence=MockParamConfidence(oos_performance=0.80, confidence_score=70.0),
            performance=MockParamPerformance(),
            run_id="20260101_120000"
        )
        challenger = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=0.60, confidence_score=50.0),  # 0.60/0.80 = 0.75 < 0.80
            performance=MockParamPerformance(),
            run_id="20260108_120000"
        )

        mock_store._set_champion_params("BTCUSDT", "15m", champion)
        mock_store._set_active_params("BTCUSDT", "15m", challenger)
        version_manager._champion_run_id = "20260101_120000"
        version_manager._challenger_run_id = "20260108_120000"

        result = version_manager.compare_performance()

        assert result.should_promote is False
        assert "rollback" in result.reason.lower()

    def test_compare_performance_none_handling(self, version_manager, mock_store):
        """None 값 처리 (TypeError 방지)"""
        champion = MockOptimizedParams(
            params={"box_L": 96},
            confidence=MockParamConfidence(oos_performance=None, confidence_score=70.0),
            performance=MockParamPerformance(),
            run_id="20260101_120000"
        )
        challenger = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=0.80, confidence_score=68.0),
            performance=MockParamPerformance(),
            run_id="20260108_120000"
        )

        mock_store._set_champion_params("BTCUSDT", "15m", champion)
        mock_store._set_active_params("BTCUSDT", "15m", challenger)
        version_manager._champion_run_id = "20260101_120000"
        version_manager._challenger_run_id = "20260108_120000"

        # TypeError 없이 실행되어야 함
        result = version_manager.compare_performance()
        assert result is not None


# ============================================================
# 승격/롤백 로직 테스트
# ============================================================

class TestPromoteRollback:
    """승격/롤백 동작 테스트"""

    def test_promote_challenger_success(self, version_manager, mock_store):
        """Challenger 승격 성공"""
        challenger = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=0.85, confidence_score=75.0),
            performance=MockParamPerformance(),
            run_id="20260108_120000"
        )
        mock_store._set_active_params("BTCUSDT", "15m", challenger)
        version_manager._challenger_run_id = "20260108_120000"

        result = version_manager.promote_challenger()

        assert result is True
        assert version_manager.champion_run_id == "20260108_120000"
        assert version_manager.challenger_run_id is None

    def test_promote_challenger_old_champion_retired(self, version_manager, mock_store):
        """승격 시 기존 Champion이 retired로 이동"""
        old_champion = MockOptimizedParams(
            params={"box_L": 96},
            confidence=MockParamConfidence(oos_performance=0.80, confidence_score=70.0),
            performance=MockParamPerformance(),
            run_id="20260101_120000"
        )
        challenger = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=0.85, confidence_score=75.0),
            performance=MockParamPerformance(),
            run_id="20260108_120000"
        )

        mock_store._set_champion_params("BTCUSDT", "15m", old_champion)
        mock_store._set_active_params("BTCUSDT", "15m", challenger)
        version_manager._champion_run_id = "20260101_120000"
        version_manager._challenger_run_id = "20260108_120000"

        version_manager.promote_challenger()

        history = version_manager.get_version_history()
        retired = [h for h in history if h["role"] == "retired" and h["run_id"] == "20260101_120000"]
        assert len(retired) == 1

    def test_rollback_to_champion_success(self, version_manager, mock_store):
        """Challenger 롤백 성공"""
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

        result = version_manager.rollback_to_champion()

        assert result is True
        assert version_manager.challenger_run_id is None
        assert version_manager.champion_run_id == "20260101_120000"  # Champion 유지

    def test_rollback_challenger_retired_in_history(self, version_manager, mock_store):
        """롤백된 Challenger가 히스토리에 retired로 기록"""
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

        version_manager.rollback_to_champion()

        history = version_manager.get_version_history()
        retired = [h for h in history if h["role"] == "retired" and h["run_id"] == "20260108_120000"]
        assert len(retired) == 1


# ============================================================
# auto_decide 테스트
# ============================================================

class TestAutoDecide:
    """auto_decide 자동 결정 로직 테스트"""

    def test_auto_decide_no_challenger(self, version_manager):
        """Challenger 없으면 no_challenger 반환"""
        action, comparison = version_manager.auto_decide()
        assert action == "no_challenger"

    def test_auto_decide_waiting_oos_not_measured(self, version_manager, mock_store):
        """OOS 미측정 시 waiting 반환"""
        challenger = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=None),  # OOS 미측정
            performance=MockParamPerformance(),
            run_id="20260108_120000"
        )
        mock_store._set_active_params("BTCUSDT", "15m", challenger)
        version_manager._challenger_run_id = "20260108_120000"

        action, comparison = version_manager.auto_decide()
        assert action == "waiting"

    def test_auto_decide_promoted(self, version_manager, mock_store):
        """승격 조건 충족 시 promoted 반환"""
        champion = MockOptimizedParams(
            params={"box_L": 96},
            confidence=MockParamConfidence(oos_performance=0.80, confidence_score=70.0),
            performance=MockParamPerformance(),
            run_id="20260101_120000"
        )
        challenger = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=0.82, confidence_score=68.0),
            performance=MockParamPerformance(),
            run_id="20260108_120000"
        )

        mock_store._set_champion_params("BTCUSDT", "15m", champion)
        mock_store._set_active_params("BTCUSDT", "15m", challenger)
        version_manager._champion_run_id = "20260101_120000"
        version_manager._challenger_run_id = "20260108_120000"

        action, comparison = version_manager.auto_decide()
        assert action == "promoted"

    def test_auto_decide_rolled_back(self, version_manager, mock_store):
        """롤백 조건 충족 시 rolled_back 반환"""
        champion = MockOptimizedParams(
            params={"box_L": 96},
            confidence=MockParamConfidence(oos_performance=0.80, confidence_score=70.0),
            performance=MockParamPerformance(),
            run_id="20260101_120000"
        )
        challenger = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=0.60, confidence_score=50.0),  # 0.75 < 0.80
            performance=MockParamPerformance(),
            run_id="20260108_120000"
        )

        mock_store._set_champion_params("BTCUSDT", "15m", champion)
        mock_store._set_active_params("BTCUSDT", "15m", challenger)
        version_manager._champion_run_id = "20260101_120000"
        version_manager._challenger_run_id = "20260108_120000"

        action, comparison = version_manager.auto_decide()
        assert action == "rolled_back"

    def test_auto_decide_kept(self, version_manager, mock_store):
        """승격/롤백 모두 아닐 때 kept 반환"""
        champion = MockOptimizedParams(
            params={"box_L": 96},
            confidence=MockParamConfidence(oos_performance=0.80, confidence_score=70.0),
            performance=MockParamPerformance(),
            run_id="20260101_120000"
        )
        # OOS ratio = 0.72/0.80 = 0.90 (롤백 0.80 이상, 승격 0.95 미만)
        challenger = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=0.72, confidence_score=60.0),
            performance=MockParamPerformance(),
            run_id="20260108_120000"
        )

        mock_store._set_champion_params("BTCUSDT", "15m", champion)
        mock_store._set_active_params("BTCUSDT", "15m", challenger)
        version_manager._champion_run_id = "20260101_120000"
        version_manager._challenger_run_id = "20260108_120000"

        action, comparison = version_manager.auto_decide()
        assert action == "kept"

    def test_auto_decide_rollback_with_zero_oos(self, version_manager, mock_store):
        """v2.2.1 버그 수정: challenger_oos=0.0일 때 롤백 트리거"""
        champion = MockOptimizedParams(
            params={"box_L": 96},
            confidence=MockParamConfidence(oos_performance=0.80, confidence_score=70.0),
            performance=MockParamPerformance(),
            run_id="20260101_120000"
        )
        # challenger_oos = 0.0 (ratio = 0.0 < 0.80 → 롤백)
        challenger = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=0.0, confidence_score=50.0),
            performance=MockParamPerformance(),
            run_id="20260108_120000"
        )

        mock_store._set_champion_params("BTCUSDT", "15m", champion)
        mock_store._set_active_params("BTCUSDT", "15m", challenger)
        version_manager._champion_run_id = "20260101_120000"
        version_manager._challenger_run_id = "20260108_120000"

        action, comparison = version_manager.auto_decide()
        # v2.2.1 수정: 0.0도 유효한 값으로 처리, 롤백 트리거
        assert action == "rolled_back"

    def test_auto_decide_champion_zero_oos_skips_rollback(self, version_manager, mock_store):
        """champion_oos=0.0일 때 ratio 계산 스킵 (기준 무의미)"""
        champion = MockOptimizedParams(
            params={"box_L": 96},
            confidence=MockParamConfidence(oos_performance=0.0, confidence_score=70.0),  # 0.0
            performance=MockParamPerformance(),
            run_id="20260101_120000"
        )
        challenger = MockOptimizedParams(
            params={"box_L": 100},
            confidence=MockParamConfidence(oos_performance=0.80, confidence_score=68.0),
            performance=MockParamPerformance(),
            run_id="20260108_120000"
        )

        mock_store._set_champion_params("BTCUSDT", "15m", champion)
        mock_store._set_active_params("BTCUSDT", "15m", challenger)
        version_manager._champion_run_id = "20260101_120000"
        version_manager._challenger_run_id = "20260108_120000"

        action, comparison = version_manager.auto_decide()
        # champion_oos=0이면 ratio 계산 안하고 kept
        # (실제로는 compare_performance에서 first version으로 처리될 수도 있음)
        assert action in ("kept", "promoted")


# ============================================================
# 상태 저장/로드 테스트
# ============================================================

class TestStatePersistence:
    """상태 저장/로드 테스트"""

    def test_save_and_load_state(self, version_manager, mock_store, temp_store_dir):
        """상태 저장 후 로드"""
        version_manager._champion_run_id = "20260101_120000"
        version_manager._challenger_run_id = "20260108_120000"
        version_manager._save_state()

        # 히스토리 파일 존재 확인
        history_file = temp_store_dir / "BTCUSDT_15m_version_history.json"
        assert history_file.exists()

        # 파일 내용 확인
        with open(history_file, "r") as f:
            data = json.load(f)

        assert data["champion_run_id"] == "20260101_120000"
        assert data["challenger_run_id"] == "20260108_120000"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
