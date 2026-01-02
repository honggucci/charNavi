"""
Performance Optimized Computations (P1: 슬리피지 방지)
=====================================================

FFT/Entropy/Physics 계산의 성능 최적화.

최적화 전략:
1. Numba JIT 적용: 루프/수치 계산 가속
2. 결과 캐싱: 동일 입력 재계산 방지
3. 벡터화: 배열 연산 최적화
4. 샘플링: 불필요한 고빈도 계산 제거

Usage:
    from wpcn._03_common._02_features.perf_optimized import (
        fast_fft_cycle,
        fast_hilbert_phase,
        cached_volatility_regime,
        FastDynamicEngine,
    )
"""
from __future__ import annotations
from functools import lru_cache
from typing import Tuple, Dict, Optional
import numpy as np
import hashlib
import time

# Numba JIT (선택적 import)
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator
    def jit(*args, **kwargs):
        def wrapper(fn):
            return fn
        return wrapper
    prange = range


# ============================================================
# Numba-Accelerated Core Functions
# ============================================================

@jit(nopython=True, cache=True, fastmath=True)
def _fast_log_returns(close: np.ndarray) -> np.ndarray:
    """로그 수익률 계산 (Numba 가속)"""
    n = len(close)
    returns = np.empty(n - 1)
    for i in range(n - 1):
        returns[i] = np.log(close[i + 1] / close[i] + 1e-10)
    return returns


@jit(nopython=True, cache=True, fastmath=True)
def _fast_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """롤링 평균 (Numba 가속)"""
    n = len(arr)
    result = np.empty(n)

    # 초기 윈도우 (불완전)
    for i in range(min(window, n)):
        result[i] = np.mean(arr[:i+1])

    # 전체 윈도우
    for i in range(window, n):
        result[i] = np.mean(arr[i-window+1:i+1])

    return result


@jit(nopython=True, cache=True, fastmath=True)
def _fast_rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """롤링 표준편차 (Numba 가속)"""
    n = len(arr)
    result = np.empty(n)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = arr[start:i+1]
        if len(window_data) < 2:
            result[i] = 0.0
        else:
            mean = np.mean(window_data)
            variance = np.mean((window_data - mean) ** 2)
            result[i] = np.sqrt(variance)

    return result


@jit(nopython=True, cache=True, fastmath=True)
def _fast_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range 계산 (Numba 가속)"""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))

    return tr


@jit(nopython=True, cache=True, fastmath=True)
def _fast_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ATR 계산 (Numba 가속)"""
    tr = _fast_true_range(high, low, close)
    return _fast_rolling_mean(tr, period)


@jit(nopython=True, cache=True, fastmath=True)
def _fast_sma(arr: np.ndarray, period: int) -> float:
    """단순 이동평균 (마지막 N개의 평균)"""
    n = len(arr)
    if n < period:
        return np.mean(arr)
    return np.mean(arr[-period:])


@jit(nopython=True, cache=True, fastmath=True)
def _fast_trend_score(close: np.ndarray) -> Tuple[float, float]:
    """
    추세 점수 계산 (Numba 가속)

    Returns:
        (trend_score, trend_strength)
    """
    n = len(close)
    if n < 100:
        return 0.0, 0.0

    # 이동평균 계산
    sma_short = _fast_sma(close, 20)
    sma_medium = _fast_sma(close, 50)
    sma_long = _fast_sma(close, 100)
    current_price = close[-1]

    trend_score = 0.0

    # 가격 vs 이동평균 위치
    if current_price > sma_short:
        trend_score += 0.15
    else:
        trend_score -= 0.15

    if current_price > sma_medium:
        trend_score += 0.15
    else:
        trend_score -= 0.15

    if current_price > sma_long:
        trend_score += 0.10
    else:
        trend_score -= 0.10

    # 이동평균 정렬
    if sma_short > sma_medium > sma_long:
        trend_score += 0.30
    elif sma_short < sma_medium < sma_long:
        trend_score -= 0.30
    elif sma_short > sma_medium:
        trend_score += 0.15
    elif sma_short < sma_medium:
        trend_score -= 0.15

    # 최근 모멘텀
    momentum = 0.0
    for i in range(max(0, n-21), n-1):
        momentum += np.log(close[i+1] / close[i] + 1e-10)

    if momentum > 0.02:
        trend_score += 0.30
    elif momentum > 0.01:
        trend_score += 0.15
    elif momentum < -0.02:
        trend_score -= 0.30
    elif momentum < -0.01:
        trend_score -= 0.15

    strength = min(1.0, abs(trend_score))

    return trend_score, strength


@jit(nopython=True, cache=True, fastmath=True)
def _fast_volatility_percentile(abs_returns: np.ndarray, current_vol: float) -> float:
    """변동성 분위수 계산 (Numba 가속)"""
    n = len(abs_returns)
    if n < 10:
        return 0.5

    count = 0
    for i in range(n):
        if abs_returns[i] <= current_vol:
            count += 1

    return count / n


@jit(nopython=True, cache=True, fastmath=True)
def _fast_atr_multipliers(volatility_percentile: float) -> Tuple[float, float, float]:
    """동적 ATR 배수 계산 (Numba 가속)"""
    vp = max(0.2, min(0.8, volatility_percentile))

    stop_ratio = 0.15 + 0.20 * (vp ** 0.7)
    tp1_ratio = 0.8 - 0.3 * (vp ** 0.8)
    tp2_ratio = 1.2 - 0.5 * (vp ** 0.8)

    return stop_ratio, tp1_ratio, tp2_ratio


# ============================================================
# FFT Cycle Detection (Optimized)
# ============================================================

def fast_fft_cycle(
    prices: np.ndarray,
    min_period: int = 5,
    max_period: int = 100
) -> Tuple[int, float]:
    """
    FFT 사이클 감지 (최적화 버전)

    numpy.fft 사용 (scipy보다 빠름)
    """
    n = len(prices)
    if n < max_period * 2:
        return max_period // 2, 0.0

    # 로그 수익률
    if HAS_NUMBA:
        returns = _fast_log_returns(prices)
    else:
        returns = np.diff(np.log(prices + 1e-10))

    # 트렌드 제거
    detrended = returns - np.mean(returns)

    # Hanning 윈도우
    window = np.hanning(len(detrended))
    windowed = detrended * window

    # FFT (numpy가 더 빠름)
    fft_vals = np.fft.fft(windowed)
    freqs = np.fft.fftfreq(len(windowed))

    # 파워 스펙트럼
    positive_mask = freqs > 0
    power = np.abs(fft_vals[positive_mask]) ** 2
    positive_freqs = freqs[positive_mask]

    # 주기 계산
    with np.errstate(divide='ignore'):
        periods = 1.0 / positive_freqs

    # 유효 범위 필터링
    valid_mask = (periods >= min_period) & (periods <= max_period)
    if not np.any(valid_mask):
        return max_period // 2, 0.0

    valid_periods = periods[valid_mask]
    valid_power = power[valid_mask]

    # 최대 파워 주기
    max_idx = np.argmax(valid_power)
    dominant_period = int(round(valid_periods[max_idx]))

    # 정규화된 파워
    total_power = np.sum(power)
    normalized_power = valid_power[max_idx] / (total_power + 1e-10)

    return dominant_period, float(normalized_power)


# ============================================================
# Hilbert Phase (Optimized)
# ============================================================

def fast_hilbert_phase(
    prices: np.ndarray,
    smooth_period: int = 5
) -> Tuple[float, str]:
    """
    Hilbert 위상 계산 (최적화 버전)

    scipy.signal.hilbert는 이미 최적화되어 있으므로
    전처리/후처리만 최적화
    """
    from scipy.signal import hilbert

    # 로그 수익률
    if HAS_NUMBA:
        returns = _fast_log_returns(prices)
    else:
        returns = np.diff(np.log(prices + 1e-10))

    # 롤링 평균으로 트렌드 제거
    if HAS_NUMBA:
        rolling_mean = _fast_rolling_mean(returns, smooth_period)
    else:
        import pandas as pd
        rolling_mean = pd.Series(returns).rolling(smooth_period, min_periods=1).mean().values

    detrended = returns - rolling_mean

    # Hilbert 변환
    analytic = hilbert(detrended)

    # 순간 위상
    phase = np.angle(analytic[-1])
    phase_normalized = phase % (2 * np.pi)

    # 위상 위치
    if phase_normalized < np.pi / 2:
        position = 'bottom'
    elif phase_normalized < np.pi:
        position = 'rising'
    elif phase_normalized < 3 * np.pi / 2:
        position = 'top'
    else:
        position = 'falling'

    return float(phase_normalized), position


# ============================================================
# Cached Volatility Regime
# ============================================================

class VolatilityCache:
    """변동성 계산 결과 캐싱"""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Tuple[str, float, float, float, float]] = {}
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0

    def _hash_input(self, returns: np.ndarray, current_vol: float) -> str:
        """입력 데이터 해시"""
        # 마지막 20개 수익률 + 현재 변동성으로 키 생성
        key_data = np.concatenate([returns[-20:], [current_vol]])
        return hashlib.md5(key_data.tobytes()).hexdigest()[:16]

    def get_or_compute(
        self,
        returns: np.ndarray,
        current_vol: float
    ) -> Tuple[str, float, float, float, float]:
        """
        캐시된 결과 반환 또는 계산

        Returns:
            (regime, percentile, stop_mult, tp1_mult, tp2_mult)
        """
        key = self._hash_input(returns, current_vol)

        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]

        self.miss_count += 1

        # 계산
        abs_returns = np.abs(returns[~np.isnan(returns)])

        if len(abs_returns) < 30:
            result = ('medium', 0.5, 0.25, 0.65, 0.95)
        else:
            # Numba 가속 분위수 계산
            if HAS_NUMBA:
                percentile = _fast_volatility_percentile(abs_returns, current_vol)
            else:
                from scipy import stats
                percentile = stats.percentileofscore(abs_returns, current_vol) / 100.0

            # Regime 분류
            if percentile < 0.25:
                regime = 'low'
            elif percentile < 0.50:
                regime = 'medium'
            elif percentile < 0.75:
                regime = 'high'
            else:
                regime = 'extreme'

            # ATR 배수
            if HAS_NUMBA:
                stop_mult, tp1_mult, tp2_mult = _fast_atr_multipliers(percentile)
            else:
                vp = np.clip(percentile, 0.2, 0.8)
                stop_mult = 0.15 + 0.20 * (vp ** 0.7)
                tp1_mult = 0.8 - 0.3 * (vp ** 0.8)
                tp2_mult = 1.2 - 0.5 * (vp ** 0.8)

            result = (regime, percentile, stop_mult, tp1_mult, tp2_mult)

        # 캐시 크기 관리
        if len(self.cache) >= self.max_size:
            # FIFO: 가장 오래된 항목 제거
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = result
        return result

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


# 글로벌 캐시 인스턴스
_volatility_cache = VolatilityCache()


def cached_volatility_regime(
    returns: np.ndarray,
    current_vol: float
) -> Tuple[str, float, float, float, float]:
    """캐시된 변동성 분석"""
    return _volatility_cache.get_or_compute(returns, current_vol)


# ============================================================
# Fast Dynamic Engine
# ============================================================

class FastDynamicEngine:
    """
    최적화된 동적 파라미터 엔진

    기존 DynamicParameterEngine 대비 개선점:
    1. Numba JIT 적용된 수치 계산
    2. 결과 캐싱
    3. 샘플링 최적화 (매 bar 계산 → 간격 계산)
    """

    def __init__(
        self,
        vol_lookback: int = 100,
        fft_min_period: int = 5,
        fft_max_period: int = 100,
        hilbert_smooth: int = 5,
        analysis_interval: int = 10  # 몇 bar마다 재계산
    ):
        self.vol_lookback = vol_lookback
        self.fft_min_period = fft_min_period
        self.fft_max_period = fft_max_period
        self.hilbert_smooth = hilbert_smooth
        self.analysis_interval = analysis_interval

        # 캐시
        self._last_result = None
        self._last_index = -1

    def compute(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        current_idx: int = -1,
        force_recompute: bool = False
    ) -> Dict:
        """
        동적 파라미터 계산 (최적화)

        Returns:
            Dict with dynamic parameters
        """
        if current_idx == -1:
            current_idx = len(close) - 1

        # 캐시 체크
        if (not force_recompute and
            self._last_result is not None and
            abs(current_idx - self._last_index) < self.analysis_interval):
            return self._last_result

        # === 1. 로그 수익률 ===
        if HAS_NUMBA and len(close) > 1:
            returns = _fast_log_returns(close[:current_idx+1])
        else:
            returns = np.diff(np.log(close[:current_idx+1] + 1e-10))

        # === 2. 현재 변동성 (ATR 기반) ===
        if HAS_NUMBA and len(high) >= 14:
            atr_values = _fast_atr(
                high[:current_idx+1],
                low[:current_idx+1],
                close[:current_idx+1],
                14
            )
            current_vol = atr_values[-1] / close[current_idx] if close[current_idx] > 0 else 0.01
        else:
            # Fallback
            current_vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.01

        # === 3. 변동성 분석 (캐시됨) ===
        recent_returns = returns[-self.vol_lookback:] if len(returns) >= self.vol_lookback else returns
        regime, percentile, stop_mult, tp1_mult, tp2_mult = cached_volatility_regime(
            recent_returns, current_vol
        )

        # === 4. 사이클 분석 (FFT) ===
        dominant_cycle, cycle_power = fast_fft_cycle(
            close[:current_idx+1],
            self.fft_min_period,
            self.fft_max_period
        )
        optimal_hold = max(5, min(dominant_cycle // 2, 50))

        # === 5. 위상 분석 (Hilbert) ===
        current_phase, phase_position = fast_hilbert_phase(
            close[:current_idx+1],
            self.hilbert_smooth
        )

        # === 6. 추세 분석 (Numba 가속) ===
        if HAS_NUMBA and len(close) >= 100:
            trend_score, trend_strength = _fast_trend_score(close[:current_idx+1])
        else:
            # Fallback
            trend_score, trend_strength = 0.0, 0.0

        if trend_score > 0.25:
            trend_direction = 'bullish'
        elif trend_score < -0.25:
            trend_direction = 'bearish'
        else:
            trend_direction = 'neutral'

        # === 7. 신뢰도 ===
        data_conf = min(1.0, len(close) / 200)
        cycle_conf = min(1.0, cycle_power * 10)
        trend_conf = trend_strength
        confidence = 0.4 * data_conf + 0.3 * cycle_conf + 0.3 * trend_conf

        # === 8. 위상 기반 조정 ===
        if phase_position in ['bottom', 'top']:
            stop_mult *= 0.8
            tp1_mult *= 1.2
        elif phase_position in ['rising', 'falling']:
            stop_mult *= 1.1
            tp1_mult *= 0.9

        result = {
            'atr_mult_stop': round(stop_mult, 3),
            'atr_mult_tp1': round(tp1_mult, 3),
            'atr_mult_tp2': round(tp2_mult, 3),
            'optimal_hold_bars': optimal_hold,
            'volatility_regime': regime,
            'volatility_percentile': round(percentile, 3),
            'dominant_cycle': dominant_cycle,
            'current_phase': round(current_phase, 3),
            'phase_position': phase_position,
            'trend_direction': trend_direction,
            'trend_strength': round(trend_strength, 3),
            'confidence': round(confidence, 3),
        }

        # 캐시 업데이트
        self._last_result = result
        self._last_index = current_idx

        return result


# ============================================================
# Performance Monitoring
# ============================================================

class PerfMonitor:
    """성능 모니터링 (프로파일링용)"""

    def __init__(self):
        self.timings: Dict[str, list] = {}

    def measure(self, name: str):
        """컨텍스트 매니저로 시간 측정"""
        class Timer:
            def __init__(self, monitor, name):
                self.monitor = monitor
                self.name = name
                self.start = None

            def __enter__(self):
                self.start = time.perf_counter()
                return self

            def __exit__(self, *args):
                elapsed = time.perf_counter() - self.start
                if self.name not in self.monitor.timings:
                    self.monitor.timings[self.name] = []
                self.monitor.timings[self.name].append(elapsed)

        return Timer(self, name)

    def report(self) -> Dict:
        """성능 리포트"""
        report = {}
        for name, times in self.timings.items():
            report[name] = {
                'count': len(times),
                'total_ms': sum(times) * 1000,
                'avg_ms': (sum(times) / len(times)) * 1000 if times else 0,
                'max_ms': max(times) * 1000 if times else 0,
            }
        return report


# 글로벌 모니터
_perf_monitor = PerfMonitor()


def get_perf_monitor() -> PerfMonitor:
    return _perf_monitor


__all__ = [
    # Numba accelerated
    "_fast_log_returns",
    "_fast_rolling_mean",
    "_fast_rolling_std",
    "_fast_true_range",
    "_fast_atr",
    "_fast_trend_score",
    "_fast_volatility_percentile",
    "_fast_atr_multipliers",
    # Optimized functions
    "fast_fft_cycle",
    "fast_hilbert_phase",
    "cached_volatility_regime",
    # Classes
    "FastDynamicEngine",
    "VolatilityCache",
    "PerfMonitor",
    "get_perf_monitor",
    # Flags
    "HAS_NUMBA",
]
