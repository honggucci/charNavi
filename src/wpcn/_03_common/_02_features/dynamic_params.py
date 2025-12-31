"""
Dynamic Parameter Optimization System

이론적 기반:
1. Maxwell-Boltzmann Distribution - 변동성 분포 모델링
2. FFT (Fast Fourier Transform) - 주기 감지
3. Hilbert Transform - 순간 위상/진폭 분석
4. Black-Scholes inspired IV estimation - 내재 변동성

각 코인/타임프레임에 맞는 동적 파라미터 생성
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq


@dataclass
class DynamicParams:
    """동적으로 계산된 트레이딩 파라미터"""
    # ATR 배수 (손절/익절용)
    atr_mult_stop: float      # 손절 ATR 배수
    atr_mult_tp1: float       # 1차 익절 ATR 배수
    atr_mult_tp2: float       # 2차 익절 ATR 배수

    # 보유 기간
    optimal_hold_bars: int    # 최적 보유 봉 수

    # 변동성 상태
    volatility_regime: str    # 'low', 'medium', 'high', 'extreme'
    volatility_percentile: float  # 현재 변동성 분위수 (0-1)

    # 사이클 정보
    dominant_cycle: int       # 지배적 사이클 길이 (봉 수)
    current_phase: float      # 현재 위상 (0-2π)
    phase_position: str       # 'bottom', 'rising', 'top', 'falling'

    # 추세 방향 (핵심!)
    trend_direction: str      # 'bullish', 'bearish', 'neutral'
    trend_strength: float     # 추세 강도 (0-1)

    # 신뢰도
    confidence: float         # 파라미터 신뢰도 (0-1)


class MaxwellBoltzmannVolatility:
    """
    Maxwell-Boltzmann 분포를 이용한 변동성 모델링

    금융시장의 수익률 분포는 정규분포보다 MB 분포에 가까움:
    - 양의 값만 가짐 (변동성은 항상 양수)
    - Fat tail 특성
    - 비대칭성

    MB 분포: f(v) = sqrt(2/π) * (v²/a³) * exp(-v²/(2a²))
    여기서 a = sqrt(kT/m) ≈ 평균 변동성
    """

    def __init__(self, lookback: int = 100):
        self.lookback = lookback

    def fit_volatility_distribution(self, returns: np.ndarray) -> Tuple[float, float, float]:
        """
        수익률로부터 MB 분포 파라미터 추정

        Returns:
            (scale_param, loc_param, shape_param)
        """
        # 절대 수익률 (변동성의 proxy)
        abs_returns = np.abs(returns[~np.isnan(returns)])

        if len(abs_returns) < 30:
            return 0.01, 0.0, 1.0

        # Maxwell 분포 fitting (scipy의 maxwell은 MB의 speed 분포)
        # scale = sqrt(kT/m), 즉 "온도" 파라미터
        try:
            params = stats.maxwell.fit(abs_returns)
            loc, scale = params

            # 분포의 모드(최빈값) = scale * sqrt(2)
            mode = scale * np.sqrt(2)

            # 분포의 평균 = scale * sqrt(8/π)
            mean = scale * np.sqrt(8 / np.pi)

            return scale, loc, mean
        except Exception as e:
            # Fallback: 단순 통계 (Maxwell fit 실패 시)
            import logging
            logging.getLogger(__name__).debug(f"Maxwell fit failed, using fallback: {e}")
            return np.std(abs_returns), 0.0, np.mean(abs_returns)

    def get_volatility_regime(self, returns: np.ndarray, current_vol: float) -> Tuple[str, float]:
        """
        현재 변동성이 어떤 regime에 있는지 판단

        MB 분포의 CDF를 이용해 현재 변동성의 분위수 계산
        """
        abs_returns = np.abs(returns[~np.isnan(returns)])

        if len(abs_returns) < 30:
            return 'medium', 0.5

        try:
            loc, scale = stats.maxwell.fit(abs_returns)[:2]
            percentile = stats.maxwell.cdf(current_vol, loc=loc, scale=scale)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Maxwell CDF failed, using percentile: {e}")
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

        return regime, percentile

    def calculate_dynamic_atr_multipliers(self, volatility_percentile: float) -> Tuple[float, float, float]:
        """
        변동성 분위수에 따른 동적 ATR 배수 계산

        원리:
        - 저변동성: 좁은 손절, 넓은 익절 (추세 지속 기대)
        - 고변동성: 넓은 손절, 좁은 익절 (빠른 청산)

        이는 변동성의 mean-reversion 특성을 활용
        """
        # 변동성 분위수를 0.2 ~ 0.8 범위로 클램프 (극단값 방지)
        vp = np.clip(volatility_percentile, 0.2, 0.8)

        # 손절 배수: 저변동성 0.3, 고변동성 0.8
        # 비선형 함수로 중간값에서 안정적
        stop_mult = 0.3 + 0.5 * (vp ** 0.7)

        # TP1 배수: 저변동성 1.2, 고변동성 0.5
        # 역관계: 변동성 높을수록 빨리 익절
        tp1_mult = 1.2 - 0.7 * (vp ** 0.8)

        # TP2 배수: TP1의 1.5~2배
        tp2_mult = tp1_mult * (1.5 + 0.5 * (1 - vp))

        return stop_mult, tp1_mult, tp2_mult


class FFTCycleDetector:
    """
    FFT를 이용한 가격 사이클 감지

    가격 시계열을 주파수 영역으로 변환하여
    지배적인 사이클(dominant cycle)을 추출
    """

    def __init__(self, min_period: int = 5, max_period: int = 100):
        self.min_period = min_period
        self.max_period = max_period

    def detect_dominant_cycle(self, prices: np.ndarray) -> Tuple[int, float]:
        """
        FFT로 지배적 사이클 길이 추출

        Returns:
            (cycle_length, power) - 사이클 길이(봉 수)와 파워
        """
        n = len(prices)
        if n < self.max_period * 2:
            return self.max_period // 2, 0.0

        # 가격을 수익률로 변환 (정상성 확보)
        returns = np.diff(np.log(prices + 1e-10))

        # 트렌드 제거 (detrending)
        detrended = returns - np.mean(returns)

        # 윈도우 함수 적용 (spectral leakage 감소)
        window = np.hanning(len(detrended))
        windowed = detrended * window

        # FFT 수행
        fft_vals = fft(windowed)
        freqs = fftfreq(len(windowed))

        # 파워 스펙트럼 (양의 주파수만)
        positive_freq_mask = freqs > 0
        power = np.abs(fft_vals[positive_freq_mask]) ** 2
        positive_freqs = freqs[positive_freq_mask]

        # 주기로 변환 (주파수의 역수)
        with np.errstate(divide='ignore'):
            periods = 1.0 / positive_freqs

        # 유효 범위 내 최대 파워 주기 찾기
        valid_mask = (periods >= self.min_period) & (periods <= self.max_period)

        if not np.any(valid_mask):
            return self.max_period // 2, 0.0

        valid_periods = periods[valid_mask]
        valid_power = power[valid_mask]

        max_idx = np.argmax(valid_power)
        dominant_period = int(round(valid_periods[max_idx]))
        max_power = valid_power[max_idx]

        # 파워 정규화 (전체 파워 대비)
        total_power = np.sum(power)
        normalized_power = max_power / (total_power + 1e-10)

        return dominant_period, normalized_power

    def get_multiple_cycles(self, prices: np.ndarray, top_n: int = 3) -> list:
        """
        상위 N개의 사이클 추출 (다중 주기 분석)
        """
        n = len(prices)
        if n < self.max_period * 2:
            return [(self.max_period // 2, 0.0)]

        returns = np.diff(np.log(prices + 1e-10))
        detrended = returns - np.mean(returns)
        window = np.hanning(len(detrended))
        windowed = detrended * window

        fft_vals = fft(windowed)
        freqs = fftfreq(len(windowed))

        positive_freq_mask = freqs > 0
        power = np.abs(fft_vals[positive_freq_mask]) ** 2
        positive_freqs = freqs[positive_freq_mask]

        with np.errstate(divide='ignore'):
            periods = 1.0 / positive_freqs

        valid_mask = (periods >= self.min_period) & (periods <= self.max_period)
        valid_periods = periods[valid_mask]
        valid_power = power[valid_mask]

        if len(valid_periods) == 0:
            return [(self.max_period // 2, 0.0)]

        # 상위 N개 추출
        top_indices = np.argsort(valid_power)[-top_n:][::-1]

        total_power = np.sum(power)
        cycles = []
        for idx in top_indices:
            period = int(round(valid_periods[idx]))
            norm_power = valid_power[idx] / (total_power + 1e-10)
            cycles.append((period, norm_power))

        return cycles


class HilbertPhaseAnalyzer:
    """
    Hilbert Transform을 이용한 순간 위상/진폭 분석

    해석 신호(Analytic Signal): z(t) = x(t) + i*H[x(t)]
    - 순간 진폭: |z(t)|
    - 순간 위상: arg(z(t))

    위상은 사이클 내 현재 위치를 나타냄:
    - 0 ~ π/2: 상승 초기 (bottom → rising)
    - π/2 ~ π: 상승 후기 (rising → top)
    - π ~ 3π/2: 하락 초기 (top → falling)
    - 3π/2 ~ 2π: 하락 후기 (falling → bottom)
    """

    def __init__(self, smooth_period: int = 5):
        self.smooth_period = smooth_period

    def compute_instantaneous_phase(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        순간 위상과 진폭 계산

        Returns:
            (phase, amplitude) - 각각 numpy 배열
        """
        # 로그 수익률로 정상화
        log_prices = np.log(prices + 1e-10)
        returns = np.diff(log_prices)

        # 트렌드 제거
        detrended = returns - pd.Series(returns).rolling(self.smooth_period, min_periods=1).mean().values

        # Hilbert 변환
        analytic_signal = hilbert(detrended)

        # 순간 진폭 (envelope)
        amplitude = np.abs(analytic_signal)

        # 순간 위상 (0 ~ 2π)
        phase = np.angle(analytic_signal)
        phase = np.mod(phase, 2 * np.pi)  # 0 ~ 2π로 정규화

        # 첫 번째 값 추가 (diff로 인해 하나 줄었으므로)
        amplitude = np.concatenate([[amplitude[0]], amplitude])
        phase = np.concatenate([[phase[0]], phase])

        return phase, amplitude

    def get_phase_position(self, phase: float) -> str:
        """
        위상을 사이클 위치로 변환
        """
        phase_normalized = phase % (2 * np.pi)

        if phase_normalized < np.pi / 2:
            return 'bottom'      # 바닥에서 상승 시작
        elif phase_normalized < np.pi:
            return 'rising'      # 상승 중
        elif phase_normalized < 3 * np.pi / 2:
            return 'top'         # 고점에서 하락 시작
        else:
            return 'falling'     # 하락 중

    def get_optimal_entry_phase(self, side: str) -> Tuple[float, float]:
        """
        매수/매도에 최적인 위상 범위 반환

        Returns:
            (min_phase, max_phase)
        """
        if side == 'long':
            # 롱: 바닥 근처 (3π/2 ~ π/2)
            return 3 * np.pi / 2, np.pi / 2
        else:
            # 숏: 고점 근처 (π/2 ~ 3π/2)
            return np.pi / 2, 3 * np.pi / 2


class TrendDetector:
    """
    다중 타임프레임 추세 감지기

    여러 이동평균과 모멘텀 지표를 결합하여 추세 방향과 강도를 판단
    상승장에서 숏 신호를 필터링하고, 하락장에서 롱 신호를 필터링
    """

    def __init__(self, short_period: int = 20, medium_period: int = 50, long_period: int = 100):
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period

    def detect_trend(self, close: np.ndarray) -> Tuple[str, float]:
        """
        추세 방향과 강도 감지

        Returns:
            (trend_direction, trend_strength)
            - trend_direction: 'bullish', 'bearish', 'neutral'
            - trend_strength: 0.0 ~ 1.0
        """
        n = len(close)

        if n < self.long_period:
            return 'neutral', 0.0

        # 이동평균 계산
        sma_short = np.mean(close[-self.short_period:])
        sma_medium = np.mean(close[-self.medium_period:])
        sma_long = np.mean(close[-self.long_period:])

        current_price = close[-1]

        # 추세 점수 계산 (-1 ~ +1)
        trend_score = 0.0

        # 1. 가격 vs 이동평균 위치 (40%)
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

        # 2. 이동평균 정렬 (30%)
        if sma_short > sma_medium > sma_long:
            trend_score += 0.30  # 완벽한 상승 정렬
        elif sma_short < sma_medium < sma_long:
            trend_score -= 0.30  # 완벽한 하락 정렬
        elif sma_short > sma_medium:
            trend_score += 0.15
        elif sma_short < sma_medium:
            trend_score -= 0.15

        # 3. 최근 모멘텀 (30%)
        recent_returns = np.diff(np.log(close[-21:] + 1e-10))  # 최근 20봉 수익률
        momentum = np.sum(recent_returns)

        if momentum > 0.02:  # 2% 이상 상승
            trend_score += 0.30
        elif momentum > 0.01:
            trend_score += 0.15
        elif momentum < -0.02:  # 2% 이상 하락
            trend_score -= 0.30
        elif momentum < -0.01:
            trend_score -= 0.15

        # 추세 방향 결정
        if trend_score > 0.25:
            direction = 'bullish'
        elif trend_score < -0.25:
            direction = 'bearish'
        else:
            direction = 'neutral'

        # 강도 정규화 (0 ~ 1)
        strength = min(1.0, abs(trend_score))

        return direction, strength


class DynamicParameterEngine:
    """
    통합 동적 파라미터 엔진

    모든 분석 모듈을 결합하여 최적의 동적 파라미터 생성
    """

    def __init__(
        self,
        vol_lookback: int = 100,
        fft_min_period: int = 5,
        fft_max_period: int = 100,
        hilbert_smooth: int = 5
    ):
        self.mb_vol = MaxwellBoltzmannVolatility(lookback=vol_lookback)
        self.fft_cycle = FFTCycleDetector(min_period=fft_min_period, max_period=fft_max_period)
        self.hilbert = HilbertPhaseAnalyzer(smooth_period=hilbert_smooth)
        self.trend_detector = TrendDetector()

        self.vol_lookback = vol_lookback

    def compute_dynamic_params(
        self,
        df: pd.DataFrame,
        current_idx: int = -1
    ) -> DynamicParams:
        """
        현재 시점의 동적 파라미터 계산

        Args:
            df: OHLCV DataFrame
            current_idx: 현재 인덱스 (-1이면 마지막)
        """
        if current_idx == -1:
            current_idx = len(df) - 1

        # 데이터 추출
        close = df['close'].values[:current_idx + 1]
        high = df['high'].values[:current_idx + 1]
        low = df['low'].values[:current_idx + 1]

        # 수익률 계산
        returns = np.diff(np.log(close + 1e-10))

        # 현재 변동성 (최근 ATR 기반)
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        current_vol = np.mean(tr[-14:]) / close[-1] if len(tr) >= 14 else np.std(returns[-20:])

        # === 1. 변동성 분석 (Maxwell-Boltzmann) ===
        recent_returns = returns[-self.vol_lookback:] if len(returns) >= self.vol_lookback else returns
        regime, percentile = self.mb_vol.get_volatility_regime(recent_returns, current_vol)
        stop_mult, tp1_mult, tp2_mult = self.mb_vol.calculate_dynamic_atr_multipliers(percentile)

        # === 2. 사이클 분석 (FFT) ===
        dominant_cycle, cycle_power = self.fft_cycle.detect_dominant_cycle(close)

        # 최적 보유 기간: 사이클의 1/4 ~ 1/2
        optimal_hold = max(5, min(dominant_cycle // 2, 50))

        # === 3. 위상 분석 (Hilbert) ===
        phase, amplitude = self.hilbert.compute_instantaneous_phase(close)
        current_phase = phase[-1]
        phase_position = self.hilbert.get_phase_position(current_phase)

        # === 4. 추세 분석 (핵심 추가!) ===
        trend_direction, trend_strength = self.trend_detector.detect_trend(close)

        # === 5. 신뢰도 계산 ===
        # 사이클 파워 + 데이터 충분성 + 추세 일관성 기반
        data_confidence = min(1.0, len(close) / 200)
        cycle_confidence = min(1.0, cycle_power * 10)
        trend_confidence = trend_strength  # 추세가 강할수록 신뢰도 높음
        confidence = 0.4 * data_confidence + 0.3 * cycle_confidence + 0.3 * trend_confidence

        # === 6. 위상 기반 파라미터 조정 ===
        if phase_position in ['bottom', 'top']:
            stop_mult *= 0.8
            tp1_mult *= 1.2
        elif phase_position in ['rising', 'falling']:
            stop_mult *= 1.1
            tp1_mult *= 0.9

        return DynamicParams(
            atr_mult_stop=round(stop_mult, 3),
            atr_mult_tp1=round(tp1_mult, 3),
            atr_mult_tp2=round(tp2_mult, 3),
            optimal_hold_bars=optimal_hold,
            volatility_regime=regime,
            volatility_percentile=round(percentile, 3),
            dominant_cycle=dominant_cycle,
            current_phase=round(current_phase, 3),
            phase_position=phase_position,
            trend_direction=trend_direction,
            trend_strength=round(trend_strength, 3),
            confidence=round(confidence, 3)
        )

    def compute_params_series(self, df: pd.DataFrame, min_lookback: int = 50) -> pd.DataFrame:
        """
        전체 시계열에 대한 동적 파라미터 계산

        Returns:
            DataFrame with dynamic parameters for each bar
        """
        n = len(df)
        results = []

        for i in range(min_lookback, n):
            params = self.compute_dynamic_params(df, i)
            results.append({
                'time': df.index[i],
                'atr_mult_stop': params.atr_mult_stop,
                'atr_mult_tp1': params.atr_mult_tp1,
                'atr_mult_tp2': params.atr_mult_tp2,
                'optimal_hold_bars': params.optimal_hold_bars,
                'volatility_regime': params.volatility_regime,
                'volatility_percentile': params.volatility_percentile,
                'dominant_cycle': params.dominant_cycle,
                'current_phase': params.current_phase,
                'phase_position': params.phase_position,
                'trend_direction': params.trend_direction,
                'trend_strength': params.trend_strength,
                'confidence': params.confidence
            })

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results).set_index('time')

        # 앞부분 채우기 (min_lookback 이전)
        first_row = result_df.iloc[0]
        prefix_data = []
        for i in range(min_lookback):
            prefix_data.append({
                'time': df.index[i],
                **first_row.to_dict()
            })

        if prefix_data:
            prefix_df = pd.DataFrame(prefix_data).set_index('time')
            result_df = pd.concat([prefix_df, result_df])

        return result_df


def apply_dynamic_params_to_signal(
    signal_side: str,
    base_atr: float,
    dynamic_params: DynamicParams,
    close_price: float
) -> dict:
    """
    동적 파라미터를 신호에 적용하여 손절/익절가 계산

    Args:
        signal_side: 'long' or 'short'
        base_atr: 현재 ATR 값
        dynamic_params: 동적 파라미터
        close_price: 현재 종가

    Returns:
        {'stop': float, 'tp1': float, 'tp2': float, 'hold_bars': int}
    """
    stop_dist = base_atr * dynamic_params.atr_mult_stop
    tp1_dist = base_atr * dynamic_params.atr_mult_tp1
    tp2_dist = base_atr * dynamic_params.atr_mult_tp2

    if signal_side == 'long':
        stop = close_price - stop_dist
        tp1 = close_price + tp1_dist
        tp2 = close_price + tp2_dist
    else:
        stop = close_price + stop_dist
        tp1 = close_price - tp1_dist
        tp2 = close_price - tp2_dist

    return {
        'stop': stop,
        'tp1': tp1,
        'tp2': tp2,
        'hold_bars': dynamic_params.optimal_hold_bars,
        'confidence': dynamic_params.confidence
    }
