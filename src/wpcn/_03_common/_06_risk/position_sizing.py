"""
포지션 사이징 모듈
- Kelly Criterion
- Fixed Risk (고정 비율)
- 최대 포지션 제한
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PositionSize:
    """포지션 사이즈 결과"""
    quantity: float          # 수량
    position_value: float    # 포지션 가치 (USDT)
    risk_amount: float       # 리스크 금액
    risk_pct: float          # 자본 대비 리스크 %
    leverage_used: float     # 사용 레버리지


class PositionSizer(ABC):
    """포지션 사이저 기본 클래스"""

    @abstractmethod
    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        leverage: float = 1.0
    ) -> PositionSize:
        """포지션 크기 계산"""
        pass


class FixedRisk(PositionSizer):
    """
    고정 리스크 방식
    - 한 거래당 자본의 N% 리스크
    """

    def __init__(
        self,
        risk_per_trade: float = 0.01,  # 1%
        max_position_pct: float = 0.3,  # 최대 30%
        max_leverage: float = 20.0
    ):
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage

    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        leverage: float = 1.0
    ) -> PositionSize:
        """
        포지션 크기 계산

        예시:
        - 자본: $10,000
        - 리스크: 1% = $100
        - 진입가: $50,000 / SL: $49,000 (2% 손실)
        - 포지션 = $100 / 2% = $5,000 (= 0.1 BTC)
        """
        # 레버리지 제한
        leverage = min(leverage, self.max_leverage)

        # 리스크 금액
        risk_amount = capital * self.risk_per_trade

        # 손실률 계산
        loss_pct = abs(entry_price - stop_loss) / entry_price

        if loss_pct == 0:
            loss_pct = 0.01  # 최소 1%

        # 포지션 가치 = 리스크 금액 / 손실률
        position_value = risk_amount / loss_pct

        # 최대 포지션 제한
        max_position = capital * self.max_position_pct * leverage
        position_value = min(position_value, max_position)

        # 수량 계산
        quantity = position_value / entry_price

        # 실제 사용 레버리지
        leverage_used = position_value / capital

        return PositionSize(
            quantity=quantity,
            position_value=position_value,
            risk_amount=min(risk_amount, position_value * loss_pct),
            risk_pct=self.risk_per_trade * 100,
            leverage_used=leverage_used
        )


class KellyCriterion(PositionSizer):
    """
    Kelly Criterion 방식
    - 최적 베팅 비율 계산
    - f* = (bp - q) / b
    - b = 평균 수익 / 평균 손실 (payoff ratio)
    - p = 승률, q = 패률 (1-p)
    """

    def __init__(
        self,
        win_rate: float = 0.5,
        avg_win: float = 0.02,      # 평균 수익률 2%
        avg_loss: float = 0.01,     # 평균 손실률 1%
        kelly_fraction: float = 0.25,  # Kelly의 25% 사용 (안전)
        max_position_pct: float = 0.3,
        max_leverage: float = 20.0
    ):
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage

    def update_stats(self, win_rate: float, avg_win: float, avg_loss: float):
        """통계 업데이트 (실제 거래 결과 기반)"""
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss

    def calculate_kelly_fraction(self) -> float:
        """Kelly 비율 계산"""
        if self.avg_loss == 0:
            return 0

        b = self.avg_win / self.avg_loss  # payoff ratio
        p = self.win_rate
        q = 1 - p

        kelly = (b * p - q) / b

        # 음수면 베팅하지 않음
        if kelly <= 0:
            return 0

        # fraction 적용 (안전)
        return kelly * self.kelly_fraction

    def calculate(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        leverage: float = 1.0
    ) -> PositionSize:
        """포지션 크기 계산"""
        leverage = min(leverage, self.max_leverage)

        # Kelly 비율
        kelly_pct = self.calculate_kelly_fraction()

        if kelly_pct <= 0:
            return PositionSize(
                quantity=0,
                position_value=0,
                risk_amount=0,
                risk_pct=0,
                leverage_used=0
            )

        # 손실률 계산
        loss_pct = abs(entry_price - stop_loss) / entry_price
        if loss_pct == 0:
            loss_pct = 0.01

        # 리스크 금액
        risk_amount = capital * kelly_pct

        # 포지션 가치
        position_value = risk_amount / loss_pct

        # 최대 포지션 제한
        max_position = capital * self.max_position_pct * leverage
        position_value = min(position_value, max_position)

        # 수량 계산
        quantity = position_value / entry_price
        leverage_used = position_value / capital

        return PositionSize(
            quantity=quantity,
            position_value=position_value,
            risk_amount=min(risk_amount, position_value * loss_pct),
            risk_pct=kelly_pct * 100,
            leverage_used=leverage_used
        )


class PositionManager:
    """
    포지션 관리자
    - 동시 포지션 제한
    - 일일 손실 한도
    - 상관관계 체크
    """

    def __init__(
        self,
        max_positions: int = 3,
        max_daily_loss_pct: float = 0.05,  # 5%
        max_correlated_positions: int = 2
    ):
        self.max_positions = max_positions
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_correlated_positions = max_correlated_positions
        self.current_positions = []
        self.daily_pnl = 0.0

    def can_open_position(self, symbol: str, capital: float) -> tuple[bool, str]:
        """새 포지션 오픈 가능 여부"""

        # 최대 포지션 수 체크
        if len(self.current_positions) >= self.max_positions:
            return False, f"최대 포지션 수 초과 ({self.max_positions}개)"

        # 일일 손실 한도 체크
        if abs(self.daily_pnl) >= capital * self.max_daily_loss_pct:
            return False, f"일일 손실 한도 도달 ({self.max_daily_loss_pct*100}%)"

        # 동일 심볼 체크
        if symbol in self.current_positions:
            return False, f"이미 {symbol} 포지션 보유 중"

        return True, "OK"

    def add_position(self, symbol: str):
        """포지션 추가"""
        self.current_positions.append(symbol)

    def remove_position(self, symbol: str, pnl: float):
        """포지션 제거"""
        if symbol in self.current_positions:
            self.current_positions.remove(symbol)
        self.daily_pnl += pnl

    def reset_daily(self):
        """일일 초기화"""
        self.daily_pnl = 0.0
