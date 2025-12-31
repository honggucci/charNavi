"""
MTF 피보나치 컨플루언스 감지기
Multi-Timeframe Fibonacci Confluence Detector

여러 타임프레임의 피보나치 되돌림 레벨이 겹치는 구간(컨플루언스 존)을 찾아
고신뢰 지지/저항 구간을 식별합니다.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

# 피보나치 레벨 정의
FIBONACCI_LEVELS = {
    'fib_0': 0.0,
    'fib_236': 0.236,
    'fib_382': 0.382,
    'fib_500': 0.500,
    'fib_618': 0.618,
    'fib_786': 0.786,
    'fib_886': 0.886,
    'fib_1000': 1.0,
}

# 확장 피보나치 (TP용)
FIBONACCI_EXTENSIONS = {
    'ext_1272': 1.272,
    'ext_1414': 1.414,
    'ext_1618': 1.618,
    'ext_2000': 2.0,
    'ext_2618': 2.618,
}


@dataclass
class SwingPoint:
    """스윙 고점/저점"""
    time: Any
    price: float
    swing_type: str  # 'high' or 'low'
    strength: int  # 좌우 몇 개 봉 기준
    timeframe: str


@dataclass
class FibonacciLevel:
    """피보나치 레벨"""
    price: float
    level_name: str
    level_pct: float
    swing_high: float
    swing_low: float
    timeframe: str
    direction: str  # 'up' (저점→고점) or 'down' (고점→저점)


@dataclass
class ConfluenceZone:
    """컨플루언스 존 - 여러 피보나치가 겹치는 구간"""
    price_low: float
    price_high: float
    price_mid: float
    strength: int  # 겹치는 레벨 수
    levels: List[FibonacciLevel]
    zone_type: str  # 'support' or 'resistance'
    confidence: float  # 0~1


def find_swing_points(
    df: pd.DataFrame,
    left_bars: int = 10,
    right_bars: int = 10,
    timeframe: str = '15m'
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """
    스윙 고점/저점 찾기

    Args:
        df: OHLCV 데이터
        left_bars: 왼쪽 확인 봉 수
        right_bars: 오른쪽 확인 봉 수
        timeframe: 타임프레임

    Returns:
        (swing_highs, swing_lows)
    """
    highs = df['high'].values
    lows = df['low'].values
    idx = df.index
    n = len(df)

    swing_highs = []
    swing_lows = []

    for i in range(left_bars, n - right_bars):
        # 스윙 고점 체크
        is_swing_high = True
        for j in range(i - left_bars, i + right_bars + 1):
            if j != i and highs[j] >= highs[i]:
                is_swing_high = False
                break

        if is_swing_high:
            swing_highs.append(SwingPoint(
                time=idx[i],
                price=float(highs[i]),
                swing_type='high',
                strength=left_bars + right_bars,
                timeframe=timeframe
            ))

        # 스윙 저점 체크
        is_swing_low = True
        for j in range(i - left_bars, i + right_bars + 1):
            if j != i and lows[j] <= lows[i]:
                is_swing_low = False
                break

        if is_swing_low:
            swing_lows.append(SwingPoint(
                time=idx[i],
                price=float(lows[i]),
                swing_type='low',
                strength=left_bars + right_bars,
                timeframe=timeframe
            ))

    return swing_highs, swing_lows


def calculate_fibonacci_levels(
    swing_high: SwingPoint,
    swing_low: SwingPoint,
    timeframe: str,
    include_extensions: bool = True
) -> List[FibonacciLevel]:
    """
    두 스윙 포인트 사이의 피보나치 레벨 계산
    """
    levels = []
    high_price = swing_high.price
    low_price = swing_low.price
    price_range = high_price - low_price

    if price_range <= 0:
        return levels

    # 방향 결정: 최근 스윙이 고점이면 하락 되돌림, 저점이면 상승 되돌림
    if swing_high.time > swing_low.time:
        # 상승 후 되돌림 (고점에서 하락)
        direction = 'down'
        for name, pct in FIBONACCI_LEVELS.items():
            fib_price = high_price - (price_range * pct)
            levels.append(FibonacciLevel(
                price=fib_price,
                level_name=name,
                level_pct=pct,
                swing_high=high_price,
                swing_low=low_price,
                timeframe=timeframe,
                direction=direction
            ))
    else:
        # 하락 후 되돌림 (저점에서 상승)
        direction = 'up'
        for name, pct in FIBONACCI_LEVELS.items():
            fib_price = low_price + (price_range * pct)
            levels.append(FibonacciLevel(
                price=fib_price,
                level_name=name,
                level_pct=pct,
                swing_high=high_price,
                swing_low=low_price,
                timeframe=timeframe,
                direction=direction
            ))

    # 확장 레벨 추가
    if include_extensions:
        for name, pct in FIBONACCI_EXTENSIONS.items():
            if direction == 'down':
                fib_price = high_price - (price_range * pct)
            else:
                fib_price = low_price + (price_range * pct)

            levels.append(FibonacciLevel(
                price=fib_price,
                level_name=name,
                level_pct=pct,
                swing_high=high_price,
                swing_low=low_price,
                timeframe=timeframe,
                direction=direction
            ))

    return levels


def find_confluence_zones(
    all_levels: List[FibonacciLevel],
    current_price: float,
    tolerance_pct: float = 0.3,  # 0.3% 이내면 겹침으로 판단
    min_confluence: int = 2
) -> List[ConfluenceZone]:
    """
    여러 피보나치 레벨이 겹치는 컨플루언스 존 찾기

    Args:
        all_levels: 모든 타임프레임의 피보나치 레벨
        current_price: 현재 가격
        tolerance_pct: 겹침 판단 허용 오차 (%)
        min_confluence: 최소 겹침 수

    Returns:
        컨플루언스 존 리스트
    """
    if not all_levels:
        return []

    # 가격순 정렬
    sorted_levels = sorted(all_levels, key=lambda x: x.price)

    zones = []
    used_indices = set()

    for i, base_level in enumerate(sorted_levels):
        if i in used_indices:
            continue

        # 이 레벨과 겹치는 다른 레벨들 찾기
        cluster = [base_level]
        cluster_indices = {i}

        tolerance = base_level.price * (tolerance_pct / 100)

        for j, other_level in enumerate(sorted_levels):
            if j in used_indices or j == i:
                continue

            # 가격 차이가 허용 오차 이내인지
            if abs(other_level.price - base_level.price) <= tolerance:
                cluster.append(other_level)
                cluster_indices.add(j)

        # 최소 컨플루언스 충족 시 존 생성
        if len(cluster) >= min_confluence:
            prices = [l.price for l in cluster]
            price_low = min(prices)
            price_high = max(prices)
            price_mid = sum(prices) / len(prices)

            # 지지/저항 판단
            if price_mid < current_price:
                zone_type = 'support'
            else:
                zone_type = 'resistance'

            # 신뢰도 계산
            # - 겹침 수 많을수록 높음
            # - 다양한 타임프레임 포함할수록 높음
            # - 황금비(0.618) 포함 시 가산점
            unique_timeframes = len(set(l.timeframe for l in cluster))
            has_golden = any(l.level_pct in [0.618, 0.382] for l in cluster)

            confidence = min(1.0, (
                len(cluster) * 0.15 +
                unique_timeframes * 0.2 +
                (0.15 if has_golden else 0)
            ))

            zones.append(ConfluenceZone(
                price_low=price_low,
                price_high=price_high,
                price_mid=price_mid,
                strength=len(cluster),
                levels=cluster,
                zone_type=zone_type,
                confidence=confidence
            ))

            used_indices.update(cluster_indices)

    # 신뢰도 순 정렬
    zones.sort(key=lambda z: z.confidence, reverse=True)

    return zones


def resample_to_timeframe(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    데이터를 상위 타임프레임으로 리샘플링

    Args:
        df: 원본 OHLCV 데이터 (15m 기준)
        target_tf: 목표 타임프레임 ('1h', '4h', '1d')

    Returns:
        리샘플링된 데이터프레임
    """
    # 타임프레임 매핑 (pandas 2.0+ 호환)
    tf_map = {
        '15m': '15min',
        '1h': '1h',
        '4h': '4h',
        '1d': '1D',
    }

    if target_tf not in tf_map:
        return df

    rule = tf_map[target_tf]

    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return resampled


class MTFFibonacciConfluence:
    """
    멀티 타임프레임 피보나치 컨플루언스 분석기
    """

    def __init__(
        self,
        timeframes: List[str] = None,
        swing_lookback: Dict[str, int] = None,
        confluence_tolerance: float = 0.3,
        min_confluence: int = 2
    ):
        """
        Args:
            timeframes: 분석할 타임프레임 리스트
            swing_lookback: 타임프레임별 스윙 탐지 봉 수
            confluence_tolerance: 컨플루언스 판단 허용 오차 (%)
            min_confluence: 최소 컨플루언스 수
        """
        self.timeframes = timeframes or ['15m', '1h', '4h', '1d']
        self.swing_lookback = swing_lookback or {
            '15m': 10,
            '1h': 8,
            '4h': 6,
            '1d': 5,
        }
        self.confluence_tolerance = confluence_tolerance
        self.min_confluence = min_confluence

        # 캐시
        self._swing_cache = {}
        self._fib_cache = {}

    def analyze(
        self,
        df: pd.DataFrame,
        current_price: Optional[float] = None,
        lookback_swings: int = 5
    ) -> Dict[str, Any]:
        """
        전체 MTF 피보나치 컨플루언스 분석

        Args:
            df: 기본 타임프레임 OHLCV 데이터 (15m)
            current_price: 현재 가격 (None이면 마지막 종가)
            lookback_swings: 각 타임프레임에서 사용할 최근 스윙 수

        Returns:
            분석 결과 딕셔너리
        """
        if current_price is None:
            current_price = float(df['close'].iloc[-1])

        all_swings = {'highs': [], 'lows': []}
        all_fib_levels = []

        # 각 타임프레임별 분석
        for tf in self.timeframes:
            # 리샘플링
            if tf == '15m':
                tf_df = df
            else:
                tf_df = resample_to_timeframe(df, tf)

            if len(tf_df) < 50:
                continue

            # 스윙 포인트 찾기
            lookback = self.swing_lookback.get(tf, 10)
            swing_highs, swing_lows = find_swing_points(
                tf_df,
                left_bars=lookback,
                right_bars=lookback,
                timeframe=tf
            )

            # 최근 스윙만 사용
            recent_highs = swing_highs[-lookback_swings:] if swing_highs else []
            recent_lows = swing_lows[-lookback_swings:] if swing_lows else []

            all_swings['highs'].extend(recent_highs)
            all_swings['lows'].extend(recent_lows)

            # 피보나치 레벨 계산 (가장 최근 고점-저점 쌍)
            if recent_highs and recent_lows:
                # 가장 최근 고점과 저점
                latest_high = max(recent_highs, key=lambda x: x.time)
                latest_low = max(recent_lows, key=lambda x: x.time)

                fib_levels = calculate_fibonacci_levels(
                    latest_high,
                    latest_low,
                    tf,
                    include_extensions=True
                )
                all_fib_levels.extend(fib_levels)

                # 두 번째 최근 스윙 쌍도 추가 (선택적)
                if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                    second_high = sorted(recent_highs, key=lambda x: x.time)[-2]
                    second_low = sorted(recent_lows, key=lambda x: x.time)[-2]

                    fib_levels2 = calculate_fibonacci_levels(
                        second_high,
                        second_low,
                        tf,
                        include_extensions=False
                    )
                    all_fib_levels.extend(fib_levels2)

        # 컨플루언스 존 찾기
        confluence_zones = find_confluence_zones(
            all_fib_levels,
            current_price,
            self.confluence_tolerance,
            self.min_confluence
        )

        # 지지/저항 분류
        support_zones = [z for z in confluence_zones if z.zone_type == 'support']
        resistance_zones = [z for z in confluence_zones if z.zone_type == 'resistance']

        # 가장 가까운 지지/저항 찾기
        nearest_support = None
        nearest_resistance = None

        if support_zones:
            nearest_support = max(support_zones, key=lambda z: z.price_mid)

        if resistance_zones:
            nearest_resistance = min(resistance_zones, key=lambda z: z.price_mid)

        return {
            'current_price': current_price,
            'all_fib_levels': all_fib_levels,
            'confluence_zones': confluence_zones,
            'support_zones': support_zones,
            'resistance_zones': resistance_zones,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'all_swings': all_swings,
        }

    def get_sc_bc_prediction(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        SC(Selling Climax) / BC(Buying Climax) 예측

        가장 가까운 고신뢰 컨플루언스 존을 SC/BC 예상가로 제시
        """
        current_price = analysis['current_price']
        nearest_support = analysis['nearest_support']
        nearest_resistance = analysis['nearest_resistance']

        result = {
            'current_price': current_price,
            'sc_prediction': None,
            'bc_prediction': None,
        }

        # SC 예측 (지지 구간 = 매집 저점)
        if nearest_support:
            distance_pct = (current_price - nearest_support.price_mid) / current_price * 100
            result['sc_prediction'] = {
                'price': nearest_support.price_mid,
                'price_range': (nearest_support.price_low, nearest_support.price_high),
                'distance_pct': distance_pct,
                'confidence': nearest_support.confidence,
                'strength': nearest_support.strength,
                'levels': [
                    {
                        'tf': l.timeframe,
                        'level': l.level_name,
                        'price': l.price
                    }
                    for l in nearest_support.levels
                ]
            }

        # BC 예측 (저항 구간 = 분산 고점)
        if nearest_resistance:
            distance_pct = (nearest_resistance.price_mid - current_price) / current_price * 100
            result['bc_prediction'] = {
                'price': nearest_resistance.price_mid,
                'price_range': (nearest_resistance.price_low, nearest_resistance.price_high),
                'distance_pct': distance_pct,
                'confidence': nearest_resistance.confidence,
                'strength': nearest_resistance.strength,
                'levels': [
                    {
                        'tf': l.timeframe,
                        'level': l.level_name,
                        'price': l.price
                    }
                    for l in nearest_resistance.levels
                ]
            }

        return result
