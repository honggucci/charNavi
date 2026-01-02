"""
Alerting & Monitoring Module (Priority 6)
==========================================

A/B Testing 및 스케줄러 이벤트에 대한 알림 시스템입니다.

지원 채널:
- Slack (Webhook)
- Telegram (Bot API)
- Discord (Webhook)
- 콘솔 로깅 (기본)

주요 알림 이벤트:
1. Champion 승격 (promote)
2. Challenger 롤백 (rollback)
3. 새 Challenger 등록 (register)
4. OOS 성능 확정 (oos_finalized)
5. 스케줄러 실행 완료 (scheduler_complete)
6. 에러 발생 (error)

사용법:
    from wpcn._08_tuning.alerting import AlertManager, AlertConfig

    # 설정
    config = AlertConfig(
        slack_webhook="https://hooks.slack.com/services/...",
        telegram_token="123456:ABC...",
        telegram_chat_id="-100123456789",
    )

    # 알림 전송
    manager = AlertManager(config)
    manager.notify_promotion(symbol="BTC-USDT", run_id="20260102_120000")
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
import os


class AlertLevel(Enum):
    """알림 심각도"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertEvent(Enum):
    """알림 이벤트 타입"""
    PROMOTION = "promotion"           # Champion 승격
    ROLLBACK = "rollback"             # Challenger 롤백
    CHALLENGER_REGISTERED = "challenger_registered"  # 새 Challenger 등록
    OOS_FINALIZED = "oos_finalized"   # OOS 성능 확정
    SCHEDULER_COMPLETE = "scheduler_complete"  # 스케줄러 완료
    SCHEDULER_ERROR = "scheduler_error"  # 스케줄러 에러
    DEGRADATION = "degradation"       # 성능 저하 감지
    SYMBOL_ADDED = "symbol_added"     # 새 심볼 추가


@dataclass
class AlertConfig:
    """
    알림 설정

    환경변수 또는 직접 설정 가능:
    - WPCN_SLACK_WEBHOOK: Slack Webhook URL
    - WPCN_TELEGRAM_TOKEN: Telegram Bot Token
    - WPCN_TELEGRAM_CHAT_ID: Telegram Chat ID
    - WPCN_DISCORD_WEBHOOK: Discord Webhook URL
    """
    # Slack
    slack_webhook: Optional[str] = None
    slack_channel: Optional[str] = None  # 기본: webhook 채널

    # Telegram
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # Discord
    discord_webhook: Optional[str] = None

    # 일반 설정
    enabled: bool = True
    console_logging: bool = True
    min_level: AlertLevel = AlertLevel.INFO
    include_details: bool = True

    # 로그 파일
    log_file: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "AlertConfig":
        """환경변수에서 설정 로드"""
        return cls(
            slack_webhook=os.environ.get("WPCN_SLACK_WEBHOOK"),
            telegram_token=os.environ.get("WPCN_TELEGRAM_TOKEN"),
            telegram_chat_id=os.environ.get("WPCN_TELEGRAM_CHAT_ID"),
            discord_webhook=os.environ.get("WPCN_DISCORD_WEBHOOK"),
            console_logging=os.environ.get("WPCN_ALERT_CONSOLE", "true").lower() == "true",
        )

    def has_any_channel(self) -> bool:
        """활성화된 외부 채널이 있는지"""
        return bool(self.slack_webhook or self.telegram_token or self.discord_webhook)


@dataclass
class Alert:
    """알림 메시지"""
    event: AlertEvent
    level: AlertLevel
    title: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # 컨텍스트
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    market: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "event": self.event.value,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "market": self.market,
            "run_id": self.run_id,
        }

    def format_slack(self) -> Dict[str, Any]:
        """Slack 메시지 포맷"""
        emoji = {
            AlertLevel.INFO: ":information_source:",
            AlertLevel.SUCCESS: ":white_check_mark:",
            AlertLevel.WARNING: ":warning:",
            AlertLevel.ERROR: ":x:",
            AlertLevel.CRITICAL: ":rotating_light:",
        }.get(self.level, ":bell:")

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} {self.title}"}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": self.message}
            }
        ]

        # 컨텍스트 추가
        context_parts = []
        if self.symbol:
            context_parts.append(f"*Symbol:* {self.symbol}")
        if self.timeframe:
            context_parts.append(f"*TF:* {self.timeframe}")
        if self.market:
            context_parts.append(f"*Market:* {self.market}")
        if self.run_id:
            context_parts.append(f"*Run ID:* {self.run_id}")

        if context_parts:
            blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": " | ".join(context_parts)}]
            })

        # 상세 정보
        if self.details:
            details_text = "\n".join(f"• {k}: {v}" for k, v in self.details.items())
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"```{details_text}```"}
            })

        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"_{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_"}]
        })

        return {"blocks": blocks}

    def format_telegram(self) -> str:
        """Telegram 메시지 포맷 (HTML)"""
        emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.SUCCESS: "✅",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }.get(self.level, "🔔")

        lines = [f"<b>{emoji} {self.title}</b>", "", self.message]

        if self.symbol or self.timeframe or self.market:
            context = []
            if self.symbol:
                context.append(f"Symbol: {self.symbol}")
            if self.timeframe:
                context.append(f"TF: {self.timeframe}")
            if self.market:
                context.append(f"Market: {self.market}")
            lines.append("")
            lines.append(" | ".join(context))

        if self.run_id:
            lines.append(f"<code>run_id: {self.run_id}</code>")

        if self.details:
            lines.append("")
            lines.append("<pre>")
            for k, v in self.details.items():
                lines.append(f"  {k}: {v}")
            lines.append("</pre>")

        lines.append("")
        lines.append(f"<i>{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</i>")

        return "\n".join(lines)

    def format_discord(self) -> Dict[str, Any]:
        """Discord Embed 포맷"""
        color = {
            AlertLevel.INFO: 0x3498db,     # 파랑
            AlertLevel.SUCCESS: 0x2ecc71,  # 초록
            AlertLevel.WARNING: 0xf39c12,  # 주황
            AlertLevel.ERROR: 0xe74c3c,    # 빨강
            AlertLevel.CRITICAL: 0x9b59b6, # 보라
        }.get(self.level, 0x95a5a6)

        embed = {
            "title": self.title,
            "description": self.message,
            "color": color,
            "timestamp": self.timestamp.isoformat(),
            "fields": []
        }

        if self.symbol:
            embed["fields"].append({"name": "Symbol", "value": self.symbol, "inline": True})
        if self.timeframe:
            embed["fields"].append({"name": "Timeframe", "value": self.timeframe, "inline": True})
        if self.market:
            embed["fields"].append({"name": "Market", "value": self.market, "inline": True})
        if self.run_id:
            embed["fields"].append({"name": "Run ID", "value": self.run_id, "inline": False})

        if self.details:
            details_text = "\n".join(f"**{k}:** {v}" for k, v in self.details.items())
            embed["fields"].append({"name": "Details", "value": details_text, "inline": False})

        return {"embeds": [embed]}

    def format_console(self) -> str:
        """콘솔 출력 포맷"""
        prefix = {
            AlertLevel.INFO: "[INFO]",
            AlertLevel.SUCCESS: "[SUCCESS]",
            AlertLevel.WARNING: "[WARNING]",
            AlertLevel.ERROR: "[ERROR]",
            AlertLevel.CRITICAL: "[CRITICAL]",
        }.get(self.level, "[ALERT]")

        parts = [
            f"{prefix} {self.title}",
            f"  {self.message}",
        ]

        if self.symbol:
            parts.append(f"  Symbol: {self.symbol}")
        if self.run_id:
            parts.append(f"  Run ID: {self.run_id}")

        if self.details:
            for k, v in self.details.items():
                parts.append(f"    {k}: {v}")

        return "\n".join(parts)


class AlertManager:
    """
    알림 관리자

    여러 채널(Slack, Telegram, Discord)로 알림을 전송합니다.
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Args:
            config: 알림 설정 (None이면 환경변수에서 로드)
        """
        self.config = config or AlertConfig.from_env()
        self._handlers: List[Callable[[Alert], None]] = []

        # 로그 파일 설정
        if self.config.log_file:
            self.config.log_file.parent.mkdir(parents=True, exist_ok=True)

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """커스텀 핸들러 추가"""
        self._handlers.append(handler)

    def send(self, alert: Alert) -> bool:
        """
        알림 전송

        Args:
            alert: 알림 객체

        Returns:
            전송 성공 여부
        """
        if not self.config.enabled:
            return False

        # 최소 레벨 체크
        level_order = [AlertLevel.INFO, AlertLevel.SUCCESS, AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
        if level_order.index(alert.level) < level_order.index(self.config.min_level):
            return False

        # v2.3.1: include_details=False면 details 제거
        if not self.config.include_details:
            alert = Alert(
                event=alert.event,
                level=alert.level,
                title=alert.title,
                message=alert.message,
                details=None,  # details 제거
                timestamp=alert.timestamp,
                symbol=alert.symbol,
                timeframe=alert.timeframe,
                market=alert.market,
                run_id=alert.run_id,
            )

        success = True

        # 콘솔 로깅
        if self.config.console_logging:
            print(alert.format_console())

        # 로그 파일
        if self.config.log_file:
            self._write_log(alert)

        # Slack
        if self.config.slack_webhook:
            if not self._send_slack(alert):
                success = False

        # Telegram
        if self.config.telegram_token and self.config.telegram_chat_id:
            if not self._send_telegram(alert):
                success = False

        # Discord
        if self.config.discord_webhook:
            if not self._send_discord(alert):
                success = False

        # 커스텀 핸들러
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"[AlertManager] Handler error: {e}")

        return success

    def _send_slack(self, alert: Alert) -> bool:
        """Slack 전송"""
        try:
            import urllib.request
            import urllib.error

            payload = alert.format_slack()
            if self.config.slack_channel:
                payload["channel"] = self.config.slack_channel

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.config.slack_webhook,
                data=data,
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200

        except urllib.error.URLError as e:
            print(f"[AlertManager] Slack error: {e}")
            return False
        except Exception as e:
            print(f"[AlertManager] Slack error: {e}")
            return False

    def _send_telegram(self, alert: Alert) -> bool:
        """Telegram 전송"""
        try:
            import urllib.request
            import urllib.error
            import urllib.parse

            url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.config.telegram_chat_id,
                "text": alert.format_telegram(),
                "parse_mode": "HTML",
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200

        except urllib.error.URLError as e:
            print(f"[AlertManager] Telegram error: {e}")
            return False
        except Exception as e:
            print(f"[AlertManager] Telegram error: {e}")
            return False

    def _send_discord(self, alert: Alert) -> bool:
        """Discord 전송"""
        try:
            import urllib.request
            import urllib.error

            payload = alert.format_discord()
            data = json.dumps(payload).encode("utf-8")

            req = urllib.request.Request(
                self.config.discord_webhook,
                data=data,
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status in (200, 204)

        except urllib.error.URLError as e:
            print(f"[AlertManager] Discord error: {e}")
            return False
        except Exception as e:
            print(f"[AlertManager] Discord error: {e}")
            return False

    def _write_log(self, alert: Alert) -> None:
        """로그 파일 기록"""
        try:
            with open(self.config.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert.to_dict(), default=str) + "\n")
        except Exception as e:
            print(f"[AlertManager] Log write error: {e}")

    # ========================================
    # 편의 메서드 (이벤트별 알림)
    # ========================================

    def notify_promotion(
        self,
        symbol: str,
        run_id: str,
        timeframe: str = "15m",
        market: str = "spot",
        old_champion_id: Optional[str] = None,
        performance: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Champion 승격 알림"""
        details = {}
        if old_champion_id:
            details["Previous Champion"] = old_champion_id
        if performance:
            details.update(performance)

        alert = Alert(
            event=AlertEvent.PROMOTION,
            level=AlertLevel.SUCCESS,
            title="Champion Promoted",
            message=f"Challenger `{run_id}` has been promoted to Champion for {symbol}",
            symbol=symbol,
            timeframe=timeframe,
            market=market,
            run_id=run_id,
            details=details if details else None,
        )
        return self.send(alert)

    def notify_rollback(
        self,
        symbol: str,
        challenger_id: str,
        champion_id: str,
        timeframe: str = "15m",
        market: str = "spot",
        reason: Optional[str] = None,
        performance: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Challenger 롤백 알림"""
        details = {"Champion Restored": champion_id}
        if reason:
            details["Reason"] = reason
        if performance:
            details.update(performance)

        alert = Alert(
            event=AlertEvent.ROLLBACK,
            level=AlertLevel.WARNING,
            title="Rollback Executed",
            message=f"Challenger `{challenger_id}` rolled back for {symbol}",
            symbol=symbol,
            timeframe=timeframe,
            market=market,
            run_id=challenger_id,
            details=details,
        )
        return self.send(alert)

    def notify_challenger_registered(
        self,
        symbol: str,
        run_id: str,
        timeframe: str = "15m",
        market: str = "spot",
        params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """새 Challenger 등록 알림"""
        alert = Alert(
            event=AlertEvent.CHALLENGER_REGISTERED,
            level=AlertLevel.INFO,
            title="Challenger Registered",
            message=f"New Challenger registered for {symbol}",
            symbol=symbol,
            timeframe=timeframe,
            market=market,
            run_id=run_id,
            details=params,
        )
        return self.send(alert)

    def notify_oos_finalized(
        self,
        symbol: str,
        run_id: str,
        oos_performance: float,
        timeframe: str = "15m",
        market: str = "spot",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """OOS 성능 확정 알림"""
        details = {"OOS Performance": f"{oos_performance:.2%}"}
        if metrics:
            details.update(metrics)

        level = AlertLevel.SUCCESS if oos_performance > 0 else AlertLevel.WARNING

        alert = Alert(
            event=AlertEvent.OOS_FINALIZED,
            level=level,
            title="OOS Performance Finalized",
            message=f"OOS test completed for {symbol}: {oos_performance:.2%}",
            symbol=symbol,
            timeframe=timeframe,
            market=market,
            run_id=run_id,
            details=details,
        )
        return self.send(alert)

    def notify_scheduler_complete(
        self,
        symbols: List[str],
        market: str = "spot",
        timeframe: str = "15m",
        results: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """스케줄러 완료 알림"""
        success_count = 0
        if results:
            for r in results.values():
                if isinstance(r, dict) and r.get("tuning", {}).get("status") == "success":
                    success_count += 1

        alert = Alert(
            event=AlertEvent.SCHEDULER_COMPLETE,
            level=AlertLevel.SUCCESS,
            title="Weekly Optimization Complete",
            message=f"Processed {len(symbols)} symbols, {success_count} successful",
            market=market,
            timeframe=timeframe,
            details={"Symbols": ", ".join(symbols)} if symbols else None,
        )
        return self.send(alert)

    def notify_scheduler_error(
        self,
        error: str,
        symbol: Optional[str] = None,
        market: str = "spot",
        timeframe: str = "15m",
    ) -> bool:
        """스케줄러 에러 알림"""
        alert = Alert(
            event=AlertEvent.SCHEDULER_ERROR,
            level=AlertLevel.ERROR,
            title="Scheduler Error",
            message=f"Error during optimization: {error}",
            symbol=symbol,
            market=market,
            timeframe=timeframe,
        )
        return self.send(alert)

    def notify_degradation(
        self,
        symbol: str,
        degradation_pct: float,
        timeframe: str = "15m",
        market: str = "spot",
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """성능 저하 감지 알림"""
        level = AlertLevel.CRITICAL if degradation_pct > 50 else AlertLevel.WARNING

        alert = Alert(
            event=AlertEvent.DEGRADATION,
            level=level,
            title="Performance Degradation Detected",
            message=f"{symbol} performance degraded by {degradation_pct:.1f}%",
            symbol=symbol,
            timeframe=timeframe,
            market=market,
            details=details,
        )
        return self.send(alert)


# 전역 싱글톤 (선택적 사용)
_global_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """전역 AlertManager 인스턴스 반환"""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
    return _global_alert_manager


def configure_alerts(config: AlertConfig) -> AlertManager:
    """전역 AlertManager 설정"""
    global _global_alert_manager
    _global_alert_manager = AlertManager(config)
    return _global_alert_manager
