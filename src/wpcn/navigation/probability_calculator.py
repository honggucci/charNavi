"""
확률 계산기 + RSI 다이버전스 + Stoch RSI 분석
Probability Calculator with RSI Divergence & Stoch RSI

Stoch RSI 과매수/과매도 구간의 종가 기준으로 RSI 다이버전스를 감지하고
도달 확률을 계산합니다.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class DivergenceSignal:
    """다이버전스 신호"""
    time: Any
    divergence_type: str  # 'bullish', 'bearish', 'hidden_bullish', 'hidden_bearish'
    price_point1: float
    price_point2: float
    rsi_point1: float
    rsi_point2: float
    strength: float  # 다이버전스 강도 (0~1)
    stoch_rsi_zone: str  # 'oversold', 'overbought', 'neutral'


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI 계산"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Wilder's smoothing
    for i in range(period, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_stoch_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic RSI 계산

    Returns:
        (stoch_rsi_k, stoch_rsi_d)
    """
    rsi = calculate_rsi(close, rsi_period)

    # Stochastic of RSI
    rsi_low = rsi.rolling(stoch_period).min()
    rsi_high = rsi.rolling(stoch_period).max()

    stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low + 1e-10) * 100

    # %K and %D
    stoch_rsi_k = stoch_rsi.rolling(k_period).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(d_period).mean()

    return stoch_rsi_k, stoch_rsi_d


def find_stoch_rsi_extremes(
    df: pd.DataFrame,
    stoch_rsi_k: pd.Series,
    oversold_level: float = 20,
    overbought_level: float = 80,
    lookback: int = 50
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stoch RSI 과매수/과매도 구간의 종가 기준점 찾기

    Returns:
        (oversold_points, overbought_points)
    """
    close = df['close']
    idx = df.index

    oversold_points = []
    overbought_points = []

    start_idx = max(0, len(df) - lookback)

    for i in range(start_idx, len(df)):
        if np.isnan(stoch_rsi_k.iloc[i]):
            continue

        stoch_val = stoch_rsi_k.iloc[i]

        # 과매도 구간 (Stoch RSI < 20)
        if stoch_val < oversold_level:
            # 극단값 찾기 (이전/이후보다 낮은 지점)
            is_extreme = True
            if i > 0 and stoch_rsi_k.iloc[i-1] < stoch_val:
                is_extreme = False
            if i < len(df) - 1 and stoch_rsi_k.iloc[i+1] < stoch_val:
                is_extreme = False

            if is_extreme:
                oversold_points.append({
                    'time': idx[i],
                    'index': i,
                    'price': float(close.iloc[i]),
                    'stoch_rsi': float(stoch_val),
                    'zone': 'oversold'
                })

        # 과매수 구간 (Stoch RSI > 80)
        elif stoch_val > overbought_level:
            is_extreme = True
            if i > 0 and stoch_rsi_k.iloc[i-1] > stoch_val:
                is_extreme = False
            if i < len(df) - 1 and stoch_rsi_k.iloc[i+1] > stoch_val:
                is_extreme = False

            if is_extreme:
                overbought_points.append({
                    'time': idx[i],
                    'index': i,
                    'price': float(close.iloc[i]),
                    'stoch_rsi': float(stoch_val),
                    'zone': 'overbought'
                })

    return oversold_points, overbought_points


def detect_rsi_divergence(
    df: pd.DataFrame,
    rsi: pd.Series,
    stoch_rsi_k: pd.Series,
    min_bars_between: int = 5,
    max_bars_between: int = 50
) -> List[DivergenceSignal]:
    """
    RSI 다이버전스 감지 (Stoch RSI 극단 구간 기준)

    Regular Divergence:
    - Bullish: 가격 Lower Low + RSI Higher Low (과매도에서)
    - Bearish: 가격 Higher High + RSI Lower High (과매수에서)

    Hidden Divergence:
    - Hidden Bullish: 가격 Higher Low + RSI Lower Low (추세 지속)
    - Hidden Bearish: 가격 Lower High + RSI Higher High (추세 지속)
    """
    close = df['close']
    idx = df.index

    # Stoch RSI 극단점 찾기
    oversold_points, overbought_points = find_stoch_rsi_extremes(
        df, stoch_rsi_k, lookback=max_bars_between * 2
    )

    divergences = []

    # 과매도 구간에서 Bullish Divergence 찾기
    for i in range(1, len(oversold_points)):
        point1 = oversold_points[i-1]
        point2 = oversold_points[i]

        bars_between = point2['index'] - point1['index']
        if bars_between < min_bars_between or bars_between > max_bars_between:
            continue

        price1 = point1['price']
        price2 = point2['price']
        rsi1 = float(rsi.iloc[point1['index']])
        rsi2 = float(rsi.iloc[point2['index']])

        if np.isnan(rsi1) or np.isnan(rsi2):
            continue

        # Regular Bullish: 가격 Lower Low + RSI Higher Low
        if price2 < price1 and rsi2 > rsi1:
            strength = min(1.0, abs(rsi2 - rsi1) / 20 + abs(price1 - price2) / price1 * 10)
            divergences.append(DivergenceSignal(
                time=point2['time'],
                divergence_type='bullish',
                price_point1=price1,
                price_point2=price2,
                rsi_point1=rsi1,
                rsi_point2=rsi2,
                strength=strength,
                stoch_rsi_zone='oversold'
            ))

        # Hidden Bullish: 가격 Higher Low + RSI Lower Low (추세 지속)
        elif price2 > price1 and rsi2 < rsi1:
            strength = min(1.0, abs(rsi1 - rsi2) / 20 + abs(price2 - price1) / price1 * 10)
            divergences.append(DivergenceSignal(
                time=point2['time'],
                divergence_type='hidden_bullish',
                price_point1=price1,
                price_point2=price2,
                rsi_point1=rsi1,
                rsi_point2=rsi2,
                strength=strength * 0.8,  # Hidden은 약간 낮은 강도
                stoch_rsi_zone='oversold'
            ))

    # 과매수 구간에서 Bearish Divergence 찾기
    for i in range(1, len(overbought_points)):
        point1 = overbought_points[i-1]
        point2 = overbought_points[i]

        bars_between = point2['index'] - point1['index']
        if bars_between < min_bars_between or bars_between > max_bars_between:
            continue

        price1 = point1['price']
        price2 = point2['price']
        rsi1 = float(rsi.iloc[point1['index']])
        rsi2 = float(rsi.iloc[point2['index']])

        if np.isnan(rsi1) or np.isnan(rsi2):
            continue

        # Regular Bearish: 가격 Higher High + RSI Lower High
        if price2 > price1 and rsi2 < rsi1:
            strength = min(1.0, abs(rsi1 - rsi2) / 20 + abs(price2 - price1) / price1 * 10)
            divergences.append(DivergenceSignal(
                time=point2['time'],
                divergence_type='bearish',
                price_point1=price1,
                price_point2=price2,
                rsi_point1=rsi1,
                rsi_point2=rsi2,
                strength=strength,
                stoch_rsi_zone='overbought'
            ))

        # Hidden Bearish: 가격 Lower High + RSI Higher High (추세 지속)
        elif price2 < price1 and rsi2 > rsi1:
            strength = min(1.0, abs(rsi2 - rsi1) / 20 + abs(price1 - price2) / price1 * 10)
            divergences.append(DivergenceSignal(
                time=point2['time'],
                divergence_type='hidden_bearish',
                price_point1=price1,
                price_point2=price2,
                rsi_point1=rsi1,
                rsi_point2=rsi2,
                strength=strength * 0.8,
                stoch_rsi_zone='overbought'
            ))

    return divergences


class ProbabilityCalculator:
    """
    종합 확률 계산기

    피보나치 컨플루언스 + 물리학 검증 + RSI 다이버전스를 종합하여
    목표 가격 도달 확률을 계산합니다.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        oversold_level: float = 20,
        overbought_level: float = 80
    ):
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level

    def analyze_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        모멘텀 지표 분석 (RSI, Stoch RSI)
        """
        close = df['close']

        # RSI
        rsi = calculate_rsi(close, self.rsi_period)

        # Stoch RSI
        stoch_rsi_k, stoch_rsi_d = calculate_stoch_rsi(
            close,
            self.rsi_period,
            self.stoch_period
        )

        # 현재 상태
        current_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50
        current_stoch_k = float(stoch_rsi_k.iloc[-1]) if not np.isnan(stoch_rsi_k.iloc[-1]) else 50
        current_stoch_d = float(stoch_rsi_d.iloc[-1]) if not np.isnan(stoch_rsi_d.iloc[-1]) else 50

        # 상태 판단
        if current_stoch_k < self.oversold_level:
            stoch_zone = 'oversold'
        elif current_stoch_k > self.overbought_level:
            stoch_zone = 'overbought'
        else:
            stoch_zone = 'neutral'

        # 다이버전스 감지
        divergences = detect_rsi_divergence(df, rsi, stoch_rsi_k)

        # 최근 다이버전스
        recent_divergence = divergences[-1] if divergences else None

        return {
            'rsi': rsi,
            'stoch_rsi_k': stoch_rsi_k,
            'stoch_rsi_d': stoch_rsi_d,
            'current_rsi': current_rsi,
            'current_stoch_k': current_stoch_k,
            'current_stoch_d': current_stoch_d,
            'stoch_zone': stoch_zone,
            'divergences': divergences,
            'recent_divergence': recent_divergence
        }

    def calculate_directional_probability(
        self,
        df: pd.DataFrame,
        momentum_analysis: Dict[str, Any],
        physics_metrics: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        방향별 확률 계산

        Returns:
            {'up': 상승확률, 'down': 하락확률, 'neutral': 횡보확률}
        """
        # 기본 확률 (50-50)
        up_prob = 0.5
        down_prob = 0.5

        # 1. Stoch RSI 기반 조정
        stoch_zone = momentum_analysis['stoch_zone']
        current_stoch_k = momentum_analysis['current_stoch_k']

        if stoch_zone == 'oversold':
            # 과매도 → 상승 확률 증가
            oversold_strength = (self.oversold_level - current_stoch_k) / self.oversold_level
            up_prob += 0.15 * oversold_strength
            down_prob -= 0.15 * oversold_strength
        elif stoch_zone == 'overbought':
            # 과매수 → 하락 확률 증가
            overbought_strength = (current_stoch_k - self.overbought_level) / (100 - self.overbought_level)
            down_prob += 0.15 * overbought_strength
            up_prob -= 0.15 * overbought_strength

        # 2. RSI 다이버전스 기반 조정
        recent_div = momentum_analysis['recent_divergence']
        if recent_div:
            div_type = recent_div.divergence_type
            div_strength = recent_div.strength

            if div_type in ['bullish', 'hidden_bullish']:
                up_prob += 0.2 * div_strength
                down_prob -= 0.2 * div_strength
            elif div_type in ['bearish', 'hidden_bearish']:
                down_prob += 0.2 * div_strength
                up_prob -= 0.2 * div_strength

        # 3. 물리학 지표 기반 조정 (있는 경우)
        if physics_metrics:
            momentum = physics_metrics.get('momentum', 0)
            trend_inertia = physics_metrics.get('trend_inertia', 0)

            # 모멘텀
            if momentum > 0:
                up_prob += 0.05
                down_prob -= 0.05
            elif momentum < 0:
                down_prob += 0.05
                up_prob -= 0.05

            # 추세 관성
            if trend_inertia > 0.3:
                up_prob += 0.1 * trend_inertia
                down_prob -= 0.1 * trend_inertia
            elif trend_inertia < -0.3:
                down_prob += 0.1 * abs(trend_inertia)
                up_prob -= 0.1 * abs(trend_inertia)

        # 정규화
        total = up_prob + down_prob
        up_prob = max(0.05, min(0.95, up_prob / total))
        down_prob = max(0.05, min(0.95, down_prob / total))

        # 횡보 확률 = 양쪽 확률이 비슷할 때
        prob_diff = abs(up_prob - down_prob)
        neutral_prob = max(0, 0.5 - prob_diff)

        # 재정규화
        total = up_prob + down_prob + neutral_prob
        up_prob /= total
        down_prob /= total
        neutral_prob /= total

        return {
            'up': up_prob,
            'down': down_prob,
            'neutral': neutral_prob
        }

    def calculate_target_probability(
        self,
        df: pd.DataFrame,
        target_price: float,
        confluence_confidence: float = 0.5,
        physics_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        특정 목표 가격 도달 확률 종합 계산
        """
        current_price = float(df['close'].iloc[-1])
        direction = 'up' if target_price > current_price else 'down'
        distance_pct = abs(target_price - current_price) / current_price * 100

        # 모멘텀 분석
        momentum_analysis = self.analyze_momentum_indicators(df)

        # 방향 확률
        dir_probs = self.calculate_directional_probability(df, momentum_analysis)

        # 기본 도달 확률 (방향 확률 × 거리 페널티)
        base_direction_prob = dir_probs[direction]
        distance_factor = max(0.1, 1.0 - distance_pct / 10)  # 10% 거리에서 페널티 최대

        # 컨플루언스 & 물리학 가중치 적용
        confidence_boost = (confluence_confidence * 0.3 + physics_confidence * 0.2)

        # 최종 확률
        final_prob = base_direction_prob * distance_factor * (1 + confidence_boost)
        final_prob = max(0.05, min(0.95, final_prob))

        # 예상 소요 시간
        avg_move = df['close'].pct_change().abs().rolling(20).mean().iloc[-1] * 100
        if avg_move > 0:
            estimated_bars = int(distance_pct / avg_move)
        else:
            estimated_bars = 100

        return {
            'target_price': target_price,
            'current_price': current_price,
            'direction': direction,
            'distance_pct': distance_pct,
            'probability': final_prob,
            'estimated_bars': min(estimated_bars, 500),
            'direction_probabilities': dir_probs,
            'momentum': {
                'rsi': momentum_analysis['current_rsi'],
                'stoch_k': momentum_analysis['current_stoch_k'],
                'stoch_zone': momentum_analysis['stoch_zone'],
                'recent_divergence': {
                    'type': momentum_analysis['recent_divergence'].divergence_type,
                    'strength': momentum_analysis['recent_divergence'].strength
                } if momentum_analysis['recent_divergence'] else None
            },
            'confidence_factors': {
                'confluence': confluence_confidence,
                'physics': physics_confidence,
                'distance_factor': distance_factor
            }
        }
