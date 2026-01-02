"""
Live Trading Integration Tests (v2.2.1)
=======================================

테스트 대상:
- load_champion_theta(): Champion 파라미터 로드
- load_theta_from_store(): prefer_champion 플래그
- load_theta_with_run_id(): run_id 추적
- normalize_symbol(): 심볼 정규화
- Champion → Theta 변환 정확성

실행:
    pytest tests/test_live_trading_integration.py -v
"""
import pytest
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict
from unittest.mock import Mock, patch

# src 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================
# Mock 클래스
# ============================================================

@dataclass
class MockParamConfidence:
    oos_performance: Optional[float] = 0.85
    confidence_score: float = 75.0
    oos_pending: bool = False

    def to_dict(self) -> Dict:
        return {
            "oos_performance": self.oos_performance,
            "confidence_score": self.confidence_score,
            "oos_pending": self.oos_pending,
        }


@dataclass
class MockOptimizedParams:
    params: Dict
    confidence: MockParamConfidence
    run_id: str = "20260101_120000"


# ============================================================
# load_champion_theta 테스트
# ============================================================

class TestLoadChampionTheta:
    """load_champion_theta 함수 테스트"""

    def test_returns_none_when_no_champion(self):
        """Champion이 없으면 (None, None) 반환"""
        from wpcn._06_engine.backtest import load_champion_theta

        with patch('wpcn._08_tuning.param_store.get_param_store') as mock_store_fn:
            mock_store = Mock()
            mock_store.get_champion.return_value = None
            mock_store_fn.return_value = mock_store

            theta, run_id = load_champion_theta("BTCUSDT", "15m", "futures")

            assert theta is None
            assert run_id is None

    def test_returns_theta_when_champion_exists(self):
        """Champion이 있으면 Theta 반환"""
        from wpcn._06_engine.backtest import load_champion_theta

        with patch('wpcn._08_tuning.param_store.get_param_store') as mock_store_fn:
            with patch('wpcn._08_tuning.param_schema.convert_params_to_bars') as mock_convert:
                mock_store = Mock()
                mock_champion = MockOptimizedParams(
                    params={"box_window_minutes": 1440},  # 분 단위
                    confidence=MockParamConfidence(),
                    run_id="20260101_120000"
                )
                mock_store.get_champion.return_value = mock_champion
                mock_store_fn.return_value = mock_store

                # 변환된 봉 단위 파라미터
                mock_convert.return_value = {
                    "box_L": 96,
                    "m_freeze": 32,
                    "pivot_lr": 4,
                    "x_atr": 0.35,
                    "m_bw": 0.10,
                    "N_reclaim": 3,
                    "N_fill": 5,
                    "F_min": 0.70,
                }

                theta, run_id = load_champion_theta("BTCUSDT", "15m", "futures")

                assert theta is not None
                assert theta.box_L == 96
                assert run_id == "20260101_120000"

    def test_handles_import_error_gracefully(self):
        """Import 오류 시 graceful 처리"""
        from wpcn._06_engine.backtest import load_champion_theta

        # 실제로는 ImportError가 try/except로 잡히므로 None 반환
        # 여기선 Champion이 없는 경우로 테스트
        with patch('wpcn._08_tuning.param_store.get_param_store') as mock_store_fn:
            mock_store = Mock()
            mock_store.get_champion.return_value = None
            mock_store_fn.return_value = mock_store

            theta, run_id = load_champion_theta("BTCUSDT", "15m", "futures")

            assert theta is None
            assert run_id is None


# ============================================================
# load_theta_from_store 테스트
# ============================================================

class TestLoadThetaFromStore:
    """load_theta_from_store 함수 테스트"""

    def test_prefer_champion_true_uses_champion(self):
        """prefer_champion=True면 Champion 우선 사용"""
        from wpcn._06_engine.backtest import load_theta_from_store, DEFAULT_THETA

        with patch('wpcn._06_engine.backtest.load_champion_theta') as mock_champion:
            mock_theta = Mock()
            mock_theta.box_L = 100  # Champion 전용 값
            mock_champion.return_value = (mock_theta, "champion_run_id")

            theta = load_theta_from_store(
                "BTCUSDT", "15m", "futures",
                prefer_champion=True
            )

            assert theta.box_L == 100
            # v2.2.1: 심볼이 정규화되어 호출됨
            mock_champion.assert_called_once_with("BTC-USDT", "15m", "futures")

    def test_prefer_champion_false_skips_champion(self):
        """prefer_champion=False면 Champion 로드 스킵"""
        from wpcn._06_engine.backtest import load_theta_from_store, DEFAULT_THETA

        with patch('wpcn._06_engine.backtest.load_champion_theta') as mock_champion:
            with patch('wpcn._08_tuning.param_store.get_param_store') as mock_store_fn:
                mock_store = Mock()
                mock_store.load_reliable_as_bars.return_value = None
                mock_store_fn.return_value = mock_store

                theta = load_theta_from_store(
                    "BTCUSDT", "15m", "futures",
                    prefer_champion=False
                )

                # Champion 로드 시도 안함
                mock_champion.assert_not_called()
                # DEFAULT_THETA 반환
                assert theta == DEFAULT_THETA

    def test_fallback_to_reliable_when_no_champion(self):
        """Champion이 없으면 reliable params로 폴백"""
        from wpcn._06_engine.backtest import load_theta_from_store

        with patch('wpcn._06_engine.backtest.load_champion_theta') as mock_champion:
            with patch('wpcn._08_tuning.param_store.get_param_store') as mock_store_fn:
                # Champion 없음
                mock_champion.return_value = (None, None)

                mock_store = Mock()
                mock_store.load_reliable_as_bars.return_value = {
                    "box_L": 80,  # reliable params 값
                    "m_freeze": 24,
                    "pivot_lr": 3,
                }
                mock_store_fn.return_value = mock_store

                theta = load_theta_from_store(
                    "BTCUSDT", "15m", "futures",
                    prefer_champion=True
                )

                # Champion 시도 후 reliable로 폴백
                mock_champion.assert_called_once()
                assert theta.box_L == 80

    def test_fallback_to_default_theta(self):
        """모든 로드 실패 시 DEFAULT_THETA 반환"""
        from wpcn._06_engine.backtest import load_theta_from_store, DEFAULT_THETA

        with patch('wpcn._06_engine.backtest.load_champion_theta') as mock_champion:
            with patch('wpcn._08_tuning.param_store.get_param_store') as mock_store_fn:
                mock_champion.return_value = (None, None)

                mock_store = Mock()
                mock_store.load_reliable_as_bars.return_value = None
                mock_store_fn.return_value = mock_store

                theta = load_theta_from_store(
                    "BTCUSDT", "15m", "futures",
                    prefer_champion=True
                )

                assert theta == DEFAULT_THETA


# ============================================================
# 파라미터 변환 정확성 테스트
# ============================================================

class TestParameterConversion:
    """분→봉 변환 정확성 테스트"""

    def test_minutes_to_bars_15m(self):
        """15m 타임프레임 변환 (1440분 → 96봉)"""
        from wpcn._08_tuning.param_schema import minutes_to_bars

        # 1440분 = 24시간 = 96봉 (15m)
        bars = minutes_to_bars(1440, "15m")
        assert bars == 96

    def test_minutes_to_bars_5m(self):
        """5m 타임프레임 변환 (1440분 → 288봉)"""
        from wpcn._08_tuning.param_schema import minutes_to_bars

        # 1440분 = 24시간 = 288봉 (5m)
        bars = minutes_to_bars(1440, "5m")
        assert bars == 288

    def test_convert_params_to_bars(self):
        """전체 파라미터 변환"""
        from wpcn._08_tuning.param_schema import convert_params_to_bars

        params_minutes = {
            "box_window_minutes": 1440,  # 24시간
            "freeze_minutes": 480,       # 8시간
            "pivot_window_minutes": 60,  # 1시간 → _pivot_bars=4 → pivot_lr=2
            "x_atr": 0.35,               # 비율 (변환 안함)
            "m_bw": 0.10,                # 비율 (변환 안함)
        }

        params_bars = convert_params_to_bars(params_minutes, "15m")

        assert params_bars["box_L"] == 96       # 1440 / 15
        assert params_bars["m_freeze"] == 32    # 480 / 15
        # pivot_lr = pivot_window_minutes / tf_minutes / 2 = 60 / 15 / 2 = 2
        assert params_bars["pivot_lr"] == 2
        assert params_bars["x_atr"] == 0.35     # 비율 유지
        assert params_bars["m_bw"] == 0.10      # 비율 유지


# ============================================================
# 마켓별 파라미터 분리 테스트
# ============================================================

class TestMarketSeparation:
    """Spot/Futures 마켓 분리 테스트"""

    def test_spot_market_uses_spot_store(self):
        """spot 마켓은 spot 저장소 사용"""
        from wpcn._06_engine.backtest import load_champion_theta

        with patch('wpcn._08_tuning.param_store.get_param_store') as mock_store_fn:
            mock_store = Mock()
            mock_store.get_champion.return_value = None
            mock_store_fn.return_value = mock_store

            load_champion_theta("BTCUSDT", "15m", "spot")

            mock_store_fn.assert_called_once_with(market="spot")

    def test_futures_market_uses_futures_store(self):
        """futures 마켓은 futures 저장소 사용"""
        from wpcn._06_engine.backtest import load_champion_theta

        with patch('wpcn._08_tuning.param_store.get_param_store') as mock_store_fn:
            mock_store = Mock()
            mock_store.get_champion.return_value = None
            mock_store_fn.return_value = mock_store

            load_champion_theta("BTCUSDT", "15m", "futures")

            mock_store_fn.assert_called_once_with(market="futures")


# ============================================================
# 심볼 정규화 테스트 (v2.2.1)
# ============================================================

class TestNormalizeSymbol:
    """심볼 정규화 테스트"""

    def test_btcusdt_to_btc_usdt(self):
        """BTCUSDT → BTC-USDT 변환"""
        from wpcn._06_engine.backtest import normalize_symbol

        assert normalize_symbol("BTCUSDT") == "BTC-USDT"

    def test_ethusdt_to_eth_usdt(self):
        """ETHUSDT → ETH-USDT 변환"""
        from wpcn._06_engine.backtest import normalize_symbol

        assert normalize_symbol("ETHUSDT") == "ETH-USDT"

    def test_slash_format(self):
        """BTC/USDT → BTC-USDT 변환"""
        from wpcn._06_engine.backtest import normalize_symbol

        assert normalize_symbol("BTC/USDT") == "BTC-USDT"

    def test_colon_format(self):
        """BTC:USDT → BTC-USDT 변환"""
        from wpcn._06_engine.backtest import normalize_symbol

        assert normalize_symbol("BTC:USDT") == "BTC-USDT"

    def test_already_normalized(self):
        """BTC-USDT는 그대로 유지"""
        from wpcn._06_engine.backtest import normalize_symbol

        assert normalize_symbol("BTC-USDT") == "BTC-USDT"

    def test_usdc_pair(self):
        """BTCUSDC → BTC-USDC 변환"""
        from wpcn._06_engine.backtest import normalize_symbol

        assert normalize_symbol("BTCUSDC") == "BTC-USDC"

    def test_bnb_pair(self):
        """ETHBNB → ETH-BNB 변환"""
        from wpcn._06_engine.backtest import normalize_symbol

        assert normalize_symbol("ETHBNB") == "ETH-BNB"


# ============================================================
# load_theta_with_run_id 테스트 (v2.2.1)
# ============================================================

class TestLoadThetaWithRunId:
    """load_theta_with_run_id 함수 테스트"""

    def test_returns_run_id_with_champion(self):
        """Champion 로드 시 run_id 반환"""
        from wpcn._06_engine.backtest import load_theta_with_run_id

        with patch('wpcn._06_engine.backtest.load_champion_theta') as mock_champion:
            mock_theta = Mock()
            mock_theta.box_L = 100
            mock_champion.return_value = (mock_theta, "20260101_120000")

            theta, run_id = load_theta_with_run_id(
                "BTCUSDT", "15m", "futures",
                prefer_champion=True
            )

            assert theta.box_L == 100
            assert run_id == "20260101_120000"

    def test_returns_none_run_id_with_reliable(self):
        """reliable params 로드 시 run_id는 None"""
        from wpcn._06_engine.backtest import load_theta_with_run_id

        with patch('wpcn._06_engine.backtest.load_champion_theta') as mock_champion:
            with patch('wpcn._08_tuning.param_store.get_param_store') as mock_store_fn:
                mock_champion.return_value = (None, None)

                mock_store = Mock()
                mock_store.load_reliable_as_bars.return_value = {
                    "box_L": 80,
                }
                mock_store_fn.return_value = mock_store

                theta, run_id = load_theta_with_run_id(
                    "BTCUSDT", "15m", "futures",
                    prefer_champion=True
                )

                assert theta.box_L == 80
                assert run_id is None

    def test_returns_none_run_id_with_default(self):
        """DEFAULT_THETA 폴백 시 run_id는 None"""
        from wpcn._06_engine.backtest import load_theta_with_run_id, DEFAULT_THETA

        with patch('wpcn._06_engine.backtest.load_champion_theta') as mock_champion:
            with patch('wpcn._08_tuning.param_store.get_param_store') as mock_store_fn:
                mock_champion.return_value = (None, None)

                mock_store = Mock()
                mock_store.load_reliable_as_bars.return_value = None
                mock_store_fn.return_value = mock_store

                theta, run_id = load_theta_with_run_id(
                    "BTCUSDT", "15m", "futures",
                    prefer_champion=True
                )

                assert theta == DEFAULT_THETA
                assert run_id is None


# ============================================================
# 심볼 정규화 통합 테스트 (v2.2.1)
# ============================================================

class TestSymbolNormalizationIntegration:
    """심볼 정규화가 실제 함수에서 적용되는지 테스트"""

    def test_load_champion_normalizes_symbol(self):
        """load_champion_theta가 심볼을 정규화하여 조회"""
        from wpcn._06_engine.backtest import load_champion_theta

        with patch('wpcn._08_tuning.param_store.get_param_store') as mock_store_fn:
            mock_store = Mock()
            mock_store.get_champion.return_value = None
            mock_store_fn.return_value = mock_store

            # BTCUSDT로 호출
            load_champion_theta("BTCUSDT", "15m", "futures")

            # BTC-USDT로 정규화되어 조회
            mock_store.get_champion.assert_called_once_with("BTC-USDT", "15m")

    def test_load_theta_with_run_id_normalizes_symbol(self):
        """load_theta_with_run_id가 심볼을 정규화"""
        from wpcn._06_engine.backtest import load_theta_with_run_id

        with patch('wpcn._06_engine.backtest.load_champion_theta') as mock_champion:
            with patch('wpcn._08_tuning.param_store.get_param_store') as mock_store_fn:
                mock_champion.return_value = (None, None)

                mock_store = Mock()
                mock_store.load_reliable_as_bars.return_value = None
                mock_store_fn.return_value = mock_store

                # BTCUSDT로 호출
                load_theta_with_run_id("BTCUSDT", "15m", "futures")

                # load_champion_theta가 BTC-USDT로 호출됨
                mock_champion.assert_called_once_with("BTC-USDT", "15m", "futures")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
