"""
동적 SL/TP 계산 모듈
- ATR 기반
- 변동성 조절
- 피보나치 레벨
"""

from dataclasses import dataclass
from typing import Literal, Optional, List
import numpy as np
import pandas as pd


@dataclass
class SLTPResult:
    """SL/TP 계산 결과"""
    stop_loss: float
    take_profit: float
    sl_pct: float
    tp_pct: float
    risk_reward: float
    atr: float
    volatility_regime: str  # 'low', 'medium', 'high'


class DynamicSLTP:
    """
    동적 SL/TP 계산기
    - 변동성에 따라 SL/TP 조절
    - 최소 RR 비율 보장
    """

    def __init__(
        self,
        atr_period: int = 14,
        sl_atr_mult_low: float = 1.5,    # 저변동성: ATR × 1.5
        sl_atr_mult_mid: float = 2.0,    # 중변동성: ATR × 2.0
        sl_atr_mult_high: float = 2.5,   # 고변동성: ATR × 2.5
        min_rr_ratio: float = 1.5,       # 최소 손익비
        max_sl_pct: float = 0.03,        # 최대 SL 3%
        min_sl_pct: float = 0.005        # 최소 SL 0.5%
    ):
        self.atr_period = atr_period
        self.sl_atr_mult = {
            "low": sl_atr_mult_low,
            "medium": sl_atr_mult_mid,
            "high": sl_atr_mult_high
        }
        self.min_rr_ratio = min_rr_ratio
        self.max_sl_pct = max_sl_pct
        self.min_sl_pct = min_sl_pct

    def calculate_atr(self, df: pd.DataFrame) -> float:
        """ATR 계산"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean().iloc[-1]

        return atr

    def get_volatility_regime(self, df: pd.DataFrame, lookback: int = 100) -> str:
        """
        변동성 레짐 판단
        - ATR / 가격 비율로 판단
        """
        atr = self.calculate_atr(df)
        price = df["close"].iloc[-1]

        atr_pct = atr / price

        # 히스토리컬 ATR% 계산
        if len(df) >= lookback:
            historical_atr_pct = []
            for i in range(lookback, len(df)):
                hist_atr = self.calculate_atr(df.iloc[i-self.atr_period:i])
                hist_price = df["close"].iloc[i]
                historical_atr_pct.append(hist_atr / hist_price)

            if historical_atr_pct:
                percentile = np.percentile(historical_atr_pct, [33, 66])

                if atr_pct < percentile[0]:
                    return "low"
                elif atr_pct < percentile[1]:
                    return "medium"
                else:
                    return "high"

        return "medium"

    def calculate(
        self,
        df: pd.DataFrame,
        entry_price: float,
        side: Literal["long", "short"],
        fib_levels: Optional[List[float]] = None
    ) -> SLTPResult:
        """
        SL/TP 계산

        Args:
            df: OHLCV 데이터
            entry_price: 진입가
            side: 'long' or 'short'
            fib_levels: 피보나치 레벨 리스트 (선택)
        """
        atr = self.calculate_atr(df)
        regime = self.get_volatility_regime(df)

        # ATR 배수 결정
        atr_mult = self.sl_atr_mult[regime]

        # SL 계산
        sl_distance = atr * atr_mult
        sl_pct = sl_distance / entry_price

        # SL 제한 적용
        sl_pct = max(self.min_sl_pct, min(sl_pct, self.max_sl_pct))
        sl_distance = entry_price * sl_pct

        # TP 계산 (최소 RR 보장)
        tp_distance = sl_distance * self.min_rr_ratio

        # 피보나치 레벨 활용 (있으면)
        if fib_levels:
            tp_distance = self._adjust_tp_with_fib(
                entry_price, tp_distance, side, fib_levels
            )

        # 가격 계산
        if side == "long":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        tp_pct = tp_distance / entry_price
        rr_ratio = tp_distance / sl_distance

        return SLTPResult(
            stop_loss=stop_loss,
            take_profit=take_profit,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            risk_reward=rr_ratio,
            atr=atr,
            volatility_regime=regime
        )

    def _adjust_tp_with_fib(
        self,
        entry_price: float,
        tp_distance: float,
        side: Literal["long", "short"],
        fib_levels: List[float]
    ) -> float:
        """피보나치 레벨로 TP 조정"""
        if side == "long":
            # 진입가보다 높은 레벨 중 가장 가까운 것
            targets = [l for l in fib_levels if l > entry_price]
            if targets:
                nearest = min(targets)
                fib_distance = nearest - entry_price
                # 더 유리한 TP 선택
                if fib_distance > tp_distance:
                    return fib_distance
        else:
            # 진입가보다 낮은 레벨 중 가장 가까운 것
            targets = [l for l in fib_levels if l < entry_price]
            if targets:
                nearest = max(targets)
                fib_distance = entry_price - nearest
                if fib_distance > tp_distance:
                    return fib_distance

        return tp_distance


class TrailingSLTP(DynamicSLTP):
    """
    트레일링 SL/TP
    - 수익 방향으로 SL 이동
    """

    def __init__(
        self,
        trail_activation_pct: float = 0.01,  # 1% 수익 시 트레일링 시작
        trail_distance_pct: float = 0.005,   # 0.5% 거리 유지
        **kwargs
    ):
        super().__init__(**kwargs)
        self.trail_activation_pct = trail_activation_pct
        self.trail_distance_pct = trail_distance_pct

    def update_trailing_sl(
        self,
        entry_price: float,
        current_price: float,
        current_sl: float,
        side: Literal["long", "short"]
    ) -> float:
        """
        트레일링 SL 업데이트

        Returns:
            새로운 SL 가격 (변경 없으면 현재 SL 반환)
        """
        if side == "long":
            profit_pct = (current_price - entry_price) / entry_price

            if profit_pct >= self.trail_activation_pct:
                # 트레일링 활성화
                new_sl = current_price * (1 - self.trail_distance_pct)
                return max(current_sl, new_sl)  # SL은 올라가기만

        else:  # short
            profit_pct = (entry_price - current_price) / entry_price

            if profit_pct >= self.trail_activation_pct:
                new_sl = current_price * (1 + self.trail_distance_pct)
                return min(current_sl, new_sl)  # SL은 내려가기만

        return current_sl
