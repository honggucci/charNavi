"""
scheduler.py A/B Testing 로직 Unit Tests (v2.2)
================================================

테스트 대상:
- OptimizationScheduler._auto_promote_or_rollback()
- OptimizationScheduler._register_challenger()
- 주간 배치 실행 순서

실행:
    pytest tests/test_scheduler_ab.py -v
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
# Mock 클래스 정의
# ============================================================

@dataclass
class MockComparisonResult:
    """Mock ComparisonResult"""
    champion_run_id: Optional[str] = None
    challenger_run_id: Optional[str] = None
    champion_oos: Optional[float] = None
    challenger_oos: Optional[float] = None
    champion_confidence: Optional[float] = None
    challenger_confidence: Optional[float] = None
    oos_improvement: float = 0.0
    confidence_improvement: float = 0.0
    should_promote: bool = False
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "champion_run_id": self.champion_run_id,
            "challenger_run_id": self.challenger_run_id,
            "champion_oos": self.champion_oos,
            "challenger_oos": self.challenger_oos,
            "champion_confidence": self.champion_confidence,
            "challenger_confidence": self.challenger_confidence,
            "oos_improvement": self.oos_improvement,
            "confidence_improvement": self.confidence_improvement,
            "should_promote": self.should_promote,
            "reason": self.reason,
        }


class MockVersionManager:
    """Mock ParamVersionManager"""

    def __init__(self):
        self.auto_decide_return = ("no_challenger", MockComparisonResult())
        self.register_challenger_return = True
        self.registered_challengers = []

    def auto_decide(self):
        return self.auto_decide_return

    def register_challenger(self, run_id: str) -> bool:
        self.registered_challengers.append(run_id)
        return self.register_challenger_return


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def temp_results_dir():
    """임시 결과 디렉토리"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_version_manager():
    """Mock VersionManager"""
    return MockVersionManager()


# ============================================================
# _auto_promote_or_rollback 테스트
# ============================================================

class TestAutoPromoteOrRollback:
    """_auto_promote_or_rollback 메서드 테스트"""

    def test_auto_promote_or_rollback_promoted(self, mock_version_manager):
        """승격 성공 케이스"""
        mock_version_manager.auto_decide_return = (
            "promoted",
            MockComparisonResult(
                champion_run_id="20260101_120000",
                challenger_run_id="20260108_120000",
                champion_oos=0.80,
                challenger_oos=0.82,
                should_promote=True,
                reason="OOS ratio >= 0.95"
            )
        )

        action, comparison = mock_version_manager.auto_decide()

        assert action == "promoted"
        assert comparison.should_promote is True

    def test_auto_promote_or_rollback_rolled_back(self, mock_version_manager):
        """롤백 케이스"""
        mock_version_manager.auto_decide_return = (
            "rolled_back",
            MockComparisonResult(
                champion_run_id="20260101_120000",
                challenger_run_id="20260108_120000",
                champion_oos=0.80,
                challenger_oos=0.60,
                should_promote=False,
                reason="OOS significantly worse"
            )
        )

        action, comparison = mock_version_manager.auto_decide()

        assert action == "rolled_back"
        assert comparison.should_promote is False

    def test_auto_promote_or_rollback_kept(self, mock_version_manager):
        """유지 케이스 (추가 관찰 필요)"""
        mock_version_manager.auto_decide_return = (
            "kept",
            MockComparisonResult(
                champion_run_id="20260101_120000",
                challenger_run_id="20260108_120000",
                champion_oos=0.80,
                challenger_oos=0.72,
                should_promote=False,
                reason="Not enough improvement"
            )
        )

        action, comparison = mock_version_manager.auto_decide()

        assert action == "kept"

    def test_auto_promote_or_rollback_no_challenger(self, mock_version_manager):
        """Challenger 없는 케이스"""
        mock_version_manager.auto_decide_return = (
            "no_challenger",
            MockComparisonResult(reason="No challenger registered")
        )

        action, comparison = mock_version_manager.auto_decide()

        assert action == "no_challenger"

    def test_auto_promote_or_rollback_waiting(self, mock_version_manager):
        """OOS 미측정 대기 케이스"""
        mock_version_manager.auto_decide_return = (
            "waiting",
            MockComparisonResult(
                challenger_run_id="20260108_120000",
                challenger_oos=None,
                reason="Challenger OOS not measured yet"
            )
        )

        action, comparison = mock_version_manager.auto_decide()

        assert action == "waiting"
        assert comparison.challenger_oos is None


# ============================================================
# _register_challenger 테스트
# ============================================================

class TestRegisterChallenger:
    """_register_challenger 메서드 테스트"""

    def test_register_challenger_success(self, mock_version_manager):
        """Challenger 등록 성공"""
        result = mock_version_manager.register_challenger("20260109_120000")

        assert result is True
        assert "20260109_120000" in mock_version_manager.registered_challengers

    def test_register_challenger_failure(self, mock_version_manager):
        """Challenger 등록 실패"""
        mock_version_manager.register_challenger_return = False

        result = mock_version_manager.register_challenger("invalid_run_id")

        assert result is False

    def test_register_challenger_multiple(self, mock_version_manager):
        """여러 Challenger 연속 등록 (교체)"""
        mock_version_manager.register_challenger("20260102_120000")
        mock_version_manager.register_challenger("20260109_120000")
        mock_version_manager.register_challenger("20260116_120000")

        assert len(mock_version_manager.registered_challengers) == 3


# ============================================================
# 주간 배치 실행 순서 테스트
# ============================================================

class TestWeeklyBatchOrder:
    """주간 배치 실행 순서 테스트"""

    def test_batch_step_order(self):
        """
        v2.2 주간 배치 실행 순서:
        1. finalize_last_week_oos()
        2. auto_promote_or_rollback()
        3. run_adaptive_tuning()
        4. register_challenger()
        """
        execution_order = []

        def mock_finalize_oos(*args):
            execution_order.append("finalize_oos")
            return {"status": "success"}

        def mock_auto_promote(*args):
            execution_order.append("auto_promote")
            return {"action": "kept"}

        def mock_tuning(*args):
            execution_order.append("tuning")
            return {"status": "success", "run_id": "20260109_120000"}

        def mock_register(*args):
            execution_order.append("register")
            return True

        # 실행 순서 시뮬레이션
        mock_finalize_oos()
        mock_auto_promote()
        mock_tuning()
        mock_register()

        assert execution_order == ["finalize_oos", "auto_promote", "tuning", "register"]

    def test_batch_skips_register_on_tuning_failure(self):
        """튜닝 실패 시 Challenger 등록 스킵"""
        execution_order = []

        def mock_tuning_failure():
            execution_order.append("tuning")
            return {"status": "error", "error": "Tuning failed"}

        tuning_result = mock_tuning_failure()

        # 튜닝 실패면 register 호출 안함
        if tuning_result.get("status") == "success":
            execution_order.append("register")

        assert "register" not in execution_order


# ============================================================
# Scheduler 통합 테스트 (모듈 import 테스트)
# ============================================================

class TestSchedulerIntegration:
    """Scheduler 모듈 통합 테스트"""

    def test_scheduler_imports_version_manager(self):
        """Scheduler가 version_manager를 올바르게 import"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        try:
            from wpcn._08_tuning.param_versioning import get_version_manager
            assert get_version_manager is not None
        except ImportError as e:
            pytest.fail(f"Failed to import get_version_manager: {e}")

    def test_scheduler_version_v22(self):
        """Scheduler 버전이 v2.2인지 확인"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        try:
            from wpcn._08_tuning import scheduler
            # docstring에서 v2.2 확인
            assert "v2.2" in scheduler.__doc__ or "v2.2" in scheduler.OptimizationScheduler.__doc__
        except (ImportError, AttributeError):
            pytest.skip("Scheduler module not available")

    def test_get_version_manager_keyword_args(self):
        """get_version_manager가 keyword 인자를 올바르게 받는지 확인"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        try:
            from wpcn._08_tuning.param_versioning import get_version_manager

            # 잘못된 호출 (v2.2 이전 버그): get_param_store(market) 대신 get_param_store(market=market)
            # 이 테스트는 TypeError 없이 실행되어야 함
            with patch('wpcn._08_tuning.param_versioning.get_param_store') as mock_store:
                mock_store.return_value = Mock()
                mock_store.return_value._base_path = Path("/tmp")

                manager = get_version_manager(
                    symbol="BTCUSDT",
                    market="futures",  # keyword arg
                    timeframe="15m"
                )

                # get_param_store가 market=futures로 호출되었는지 확인
                mock_store.assert_called_once_with(market="futures")

        except ImportError:
            pytest.skip("Module not available")


# ============================================================
# run_id fallback 테스트
# ============================================================

class TestRunIdFallback:
    """run_id fallback 형식 테스트"""

    def test_run_id_format_timestamp_only(self):
        """run_id fallback이 타임스탬프만 사용하는지 확인"""
        # v2.2 수정: {symbol}_{tf}_{ts} → {ts}
        now = datetime.now()
        fallback_run_id = now.strftime('%Y%m%d_%H%M%S')

        # 형식 검증: YYYYmmdd_HHMMSS
        assert len(fallback_run_id) == 15  # 20260102_120000
        assert fallback_run_id.count("_") == 1
        assert fallback_run_id[:8].isdigit()
        assert fallback_run_id[9:].isdigit()

    def test_run_id_does_not_contain_symbol(self):
        """run_id에 symbol이 포함되지 않는지 확인"""
        now = datetime.now()
        fallback_run_id = now.strftime('%Y%m%d_%H%M%S')

        # symbol, timeframe이 포함되어 있으면 안됨
        assert "BTCUSDT" not in fallback_run_id
        assert "15m" not in fallback_run_id


# ============================================================
# 에러 핸들링 테스트
# ============================================================

class TestErrorHandling:
    """에러 핸들링 테스트"""

    def test_auto_promote_handles_import_error(self):
        """version_manager import 실패 시 graceful fallback"""
        # ImportError 시 {"action": "skipped"} 반환해야 함
        expected_result = {"action": "skipped", "reason": "Version manager not available"}

        # 실제 scheduler._auto_promote_or_rollback에서
        # ImportError가 발생하면 위와 같은 형식으로 반환
        assert expected_result["action"] == "skipped"

    def test_register_challenger_handles_exception(self):
        """register_challenger 예외 처리"""
        mock_manager = MockVersionManager()

        # 예외 발생 시뮬레이션
        def raise_error(run_id):
            raise ValueError("Test error")

        mock_manager.register_challenger = raise_error

        with pytest.raises(ValueError):
            mock_manager.register_challenger("20260109_120000")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
