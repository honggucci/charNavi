"""
Alerting & Monitoring Tests (Priority 6)
=========================================

AlertManager, AlertConfig, Alert 클래스 테스트
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wpcn._08_tuning.alerting import (
    AlertManager,
    AlertConfig,
    Alert,
    AlertLevel,
    AlertEvent,
    get_alert_manager,
    configure_alerts,
)


# ========================================
# AlertConfig Tests
# ========================================

class TestAlertConfig:
    """AlertConfig 테스트"""

    def test_default_config(self):
        """기본 설정"""
        config = AlertConfig()

        assert config.enabled is True
        assert config.console_logging is True
        assert config.slack_webhook is None
        assert config.telegram_token is None
        assert config.discord_webhook is None
        assert config.min_level == AlertLevel.INFO

    def test_from_env(self):
        """환경변수에서 로드"""
        with patch.dict("os.environ", {
            "WPCN_SLACK_WEBHOOK": "https://hooks.slack.com/test",
            "WPCN_TELEGRAM_TOKEN": "123456:ABC",
            "WPCN_TELEGRAM_CHAT_ID": "-100123",
            "WPCN_DISCORD_WEBHOOK": "https://discord.com/api/webhooks/test",
        }):
            config = AlertConfig.from_env()

            assert config.slack_webhook == "https://hooks.slack.com/test"
            assert config.telegram_token == "123456:ABC"
            assert config.telegram_chat_id == "-100123"
            assert config.discord_webhook == "https://discord.com/api/webhooks/test"

    def test_has_any_channel_false(self):
        """외부 채널 없음"""
        config = AlertConfig()
        assert config.has_any_channel() is False

    def test_has_any_channel_true_slack(self):
        """Slack 채널 있음"""
        config = AlertConfig(slack_webhook="https://hooks.slack.com/test")
        assert config.has_any_channel() is True

    def test_has_any_channel_true_telegram(self):
        """Telegram 채널 있음"""
        config = AlertConfig(telegram_token="123:ABC", telegram_chat_id="-100")
        assert config.has_any_channel() is True


# ========================================
# Alert Tests
# ========================================

class TestAlert:
    """Alert 클래스 테스트"""

    def test_to_dict(self):
        """딕셔너리 변환"""
        alert = Alert(
            event=AlertEvent.PROMOTION,
            level=AlertLevel.SUCCESS,
            title="Test",
            message="Test message",
            symbol="BTC-USDT",
            run_id="20260102_120000",
        )

        d = alert.to_dict()
        assert d["event"] == "promotion"
        assert d["level"] == "success"
        assert d["title"] == "Test"
        assert d["symbol"] == "BTC-USDT"
        assert d["run_id"] == "20260102_120000"

    def test_format_slack(self):
        """Slack 포맷"""
        alert = Alert(
            event=AlertEvent.PROMOTION,
            level=AlertLevel.SUCCESS,
            title="Champion Promoted",
            message="New champion for BTC-USDT",
            symbol="BTC-USDT",
        )

        payload = alert.format_slack()
        assert "blocks" in payload
        assert len(payload["blocks"]) >= 2  # header, section, context

    def test_format_telegram(self):
        """Telegram 포맷"""
        alert = Alert(
            event=AlertEvent.ROLLBACK,
            level=AlertLevel.WARNING,
            title="Rollback",
            message="Challenger rolled back",
            symbol="ETH-USDT",
        )

        text = alert.format_telegram()
        assert "Rollback" in text
        assert "ETH-USDT" in text
        assert "<b>" in text  # HTML 포맷

    def test_format_discord(self):
        """Discord 포맷"""
        alert = Alert(
            event=AlertEvent.SCHEDULER_ERROR,
            level=AlertLevel.ERROR,
            title="Error",
            message="Something failed",
        )

        payload = alert.format_discord()
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1
        assert payload["embeds"][0]["title"] == "Error"

    def test_format_console(self):
        """콘솔 포맷"""
        alert = Alert(
            event=AlertEvent.CHALLENGER_REGISTERED,
            level=AlertLevel.INFO,
            title="Challenger Registered",
            message="New challenger",
            symbol="SOL-USDT",
            run_id="20260102_130000",
        )

        text = alert.format_console()
        assert "[INFO]" in text
        assert "Challenger Registered" in text
        assert "SOL-USDT" in text


# ========================================
# AlertManager Tests
# ========================================

class TestAlertManager:
    """AlertManager 테스트"""

    def test_init_default(self):
        """기본 초기화"""
        manager = AlertManager()
        assert manager.config is not None
        assert manager.config.console_logging is True

    def test_init_with_config(self):
        """설정으로 초기화"""
        config = AlertConfig(enabled=False, console_logging=False)
        manager = AlertManager(config)
        assert manager.config.enabled is False
        assert manager.config.console_logging is False

    def test_send_disabled(self):
        """비활성화 상태에서 전송"""
        config = AlertConfig(enabled=False)
        manager = AlertManager(config)

        alert = Alert(
            event=AlertEvent.PROMOTION,
            level=AlertLevel.SUCCESS,
            title="Test",
            message="Test",
        )

        result = manager.send(alert)
        assert result is False

    def test_send_below_min_level(self):
        """최소 레벨 미만"""
        config = AlertConfig(min_level=AlertLevel.WARNING, console_logging=False)
        manager = AlertManager(config)

        alert = Alert(
            event=AlertEvent.CHALLENGER_REGISTERED,
            level=AlertLevel.INFO,  # WARNING 미만
            title="Test",
            message="Test",
        )

        result = manager.send(alert)
        assert result is False

    def test_include_details_false_strips_details(self):
        """include_details=False면 details 제거"""
        config = AlertConfig(include_details=False, console_logging=False)
        manager = AlertManager(config)

        received = []
        manager.add_handler(lambda a: received.append(a))

        alert = Alert(
            event=AlertEvent.PROMOTION,
            level=AlertLevel.SUCCESS,
            title="Test",
            message="Test",
            details={"key": "value", "another": 123},  # details 있음
        )

        manager.send(alert)
        assert len(received) == 1
        assert received[0].details is None  # details가 제거됨

    def test_send_console_only(self, capsys):
        """콘솔 전송"""
        config = AlertConfig(console_logging=True)
        manager = AlertManager(config)

        alert = Alert(
            event=AlertEvent.PROMOTION,
            level=AlertLevel.SUCCESS,
            title="Console Test",
            message="This should print",
        )

        result = manager.send(alert)
        assert result is True

        captured = capsys.readouterr()
        assert "Console Test" in captured.out

    def test_add_handler(self):
        """커스텀 핸들러"""
        config = AlertConfig(console_logging=False)
        manager = AlertManager(config)

        received = []
        manager.add_handler(lambda a: received.append(a))

        alert = Alert(
            event=AlertEvent.OOS_FINALIZED,
            level=AlertLevel.SUCCESS,
            title="Handler Test",
            message="Test",
        )

        manager.send(alert)
        assert len(received) == 1
        assert received[0].title == "Handler Test"


# ========================================
# Convenience Method Tests
# ========================================

class TestAlertManagerConvenienceMethods:
    """편의 메서드 테스트"""

    def setup_method(self):
        """각 테스트 전 매니저 생성"""
        self.config = AlertConfig(console_logging=False)
        self.manager = AlertManager(self.config)
        self.received = []
        self.manager.add_handler(lambda a: self.received.append(a))

    def test_notify_promotion(self):
        """승격 알림"""
        self.manager.notify_promotion(
            symbol="BTC-USDT",
            run_id="20260102_120000",
            old_champion_id="20251225_100000",
        )

        assert len(self.received) == 1
        alert = self.received[0]
        assert alert.event == AlertEvent.PROMOTION
        assert alert.level == AlertLevel.SUCCESS
        assert alert.symbol == "BTC-USDT"
        assert alert.run_id == "20260102_120000"

    def test_notify_rollback(self):
        """롤백 알림"""
        self.manager.notify_rollback(
            symbol="ETH-USDT",
            challenger_id="20260102_120000",
            champion_id="20251225_100000",
            reason="OOS performance below threshold",
        )

        assert len(self.received) == 1
        alert = self.received[0]
        assert alert.event == AlertEvent.ROLLBACK
        assert alert.level == AlertLevel.WARNING
        assert "below threshold" in str(alert.details)

    def test_notify_challenger_registered(self):
        """Challenger 등록 알림"""
        self.manager.notify_challenger_registered(
            symbol="SOL-USDT",
            run_id="20260102_130000",
        )

        assert len(self.received) == 1
        alert = self.received[0]
        assert alert.event == AlertEvent.CHALLENGER_REGISTERED
        assert alert.level == AlertLevel.INFO

    def test_notify_oos_finalized_positive(self):
        """OOS 확정 알림 (양수)"""
        self.manager.notify_oos_finalized(
            symbol="BTC-USDT",
            run_id="20260102_120000",
            oos_performance=0.05,  # +5%
        )

        assert len(self.received) == 1
        alert = self.received[0]
        assert alert.event == AlertEvent.OOS_FINALIZED
        assert alert.level == AlertLevel.SUCCESS  # 양수면 SUCCESS

    def test_notify_oos_finalized_negative(self):
        """OOS 확정 알림 (음수)"""
        self.manager.notify_oos_finalized(
            symbol="BTC-USDT",
            run_id="20260102_120000",
            oos_performance=-0.02,  # -2%
        )

        assert len(self.received) == 1
        alert = self.received[0]
        assert alert.level == AlertLevel.WARNING  # 음수면 WARNING

    def test_notify_scheduler_complete(self):
        """스케줄러 완료 알림"""
        self.manager.notify_scheduler_complete(
            symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
            results={
                "BTC-USDT": {"tuning": {"status": "success"}},
                "ETH-USDT": {"tuning": {"status": "success"}},
                "SOL-USDT": {"tuning": {"status": "error"}},
            }
        )

        assert len(self.received) == 1
        alert = self.received[0]
        assert alert.event == AlertEvent.SCHEDULER_COMPLETE
        assert "2 successful" in alert.message  # 2/3 성공

    def test_notify_scheduler_error(self):
        """스케줄러 에러 알림"""
        self.manager.notify_scheduler_error(
            error="Connection timeout",
            symbol="BTC-USDT",
        )

        assert len(self.received) == 1
        alert = self.received[0]
        assert alert.event == AlertEvent.SCHEDULER_ERROR
        assert alert.level == AlertLevel.ERROR

    def test_notify_degradation_moderate(self):
        """성능 저하 알림 (중간)"""
        self.manager.notify_degradation(
            symbol="BTC-USDT",
            degradation_pct=30.0,
        )

        assert len(self.received) == 1
        alert = self.received[0]
        assert alert.event == AlertEvent.DEGRADATION
        assert alert.level == AlertLevel.WARNING  # 50% 미만

    def test_notify_degradation_severe(self):
        """성능 저하 알림 (심각)"""
        self.manager.notify_degradation(
            symbol="BTC-USDT",
            degradation_pct=60.0,
        )

        assert len(self.received) == 1
        alert = self.received[0]
        assert alert.level == AlertLevel.CRITICAL  # 50% 이상


# ========================================
# Singleton Tests
# ========================================

class TestAlertManagerSingleton:
    """전역 매니저 테스트"""

    def test_get_alert_manager(self):
        """전역 매니저 조회"""
        manager1 = get_alert_manager()
        manager2 = get_alert_manager()
        assert manager1 is manager2

    def test_configure_alerts(self):
        """전역 매니저 설정"""
        config = AlertConfig(enabled=False)
        manager = configure_alerts(config)

        assert manager.config.enabled is False
        assert get_alert_manager() is manager


# ========================================
# External Channel Tests (Mocked)
# ========================================

class TestExternalChannels:
    """외부 채널 전송 테스트 (mock)"""

    def test_send_slack_success(self):
        """Slack 전송 성공"""
        config = AlertConfig(
            slack_webhook="https://hooks.slack.com/test",
            console_logging=False,
        )
        manager = AlertManager(config)

        alert = Alert(
            event=AlertEvent.PROMOTION,
            level=AlertLevel.SUCCESS,
            title="Test",
            message="Test",
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = manager.send(alert)
            assert result is True
            mock_urlopen.assert_called_once()

    def test_send_telegram_success(self):
        """Telegram 전송 성공"""
        config = AlertConfig(
            telegram_token="123456:ABC",
            telegram_chat_id="-100123",
            console_logging=False,
        )
        manager = AlertManager(config)

        alert = Alert(
            event=AlertEvent.ROLLBACK,
            level=AlertLevel.WARNING,
            title="Test",
            message="Test",
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = manager.send(alert)
            assert result is True

    def test_send_discord_success(self):
        """Discord 전송 성공"""
        config = AlertConfig(
            discord_webhook="https://discord.com/api/webhooks/test",
            console_logging=False,
        )
        manager = AlertManager(config)

        alert = Alert(
            event=AlertEvent.SCHEDULER_COMPLETE,
            level=AlertLevel.SUCCESS,
            title="Test",
            message="Test",
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 204  # Discord returns 204
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = manager.send(alert)
            assert result is True

    def test_send_slack_failure(self):
        """Slack 전송 실패"""
        import urllib.error

        config = AlertConfig(
            slack_webhook="https://hooks.slack.com/test",
            console_logging=False,
        )
        manager = AlertManager(config)

        alert = Alert(
            event=AlertEvent.PROMOTION,
            level=AlertLevel.SUCCESS,
            title="Test",
            message="Test",
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Connection failed")

            result = manager.send(alert)
            # 실패해도 전체는 False
            assert result is False
