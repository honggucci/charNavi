"""
물리학 기반 신뢰도 검증기
Physics-Based Reliability Validator

물리학 법칙을 시장에 적용하여 가격 움직임의 신뢰성을 검증합니다.
- 모멘텀 (F = ma): 가격 가속도 분석
- 에너지 보존: 거래량 × 변동폭 = 에너지
- 파동 역학: 사이클 분석
- 엔트로피: 시장 혼란도/질서도
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq


@dataclass
class PhysicsMetrics:
    """물리학 지표"""
    momentum: float  # 모멘텀 (가격 가속도)
    energy: float  # 에너지 (거래량 × 변동폭)
    entropy: float  # 엔트로피 (시장 혼란도)
    wave_phase: float  # 파동 위상 (0~2π)
    wave_amplitude: float  # 파동 진폭
    dominant_cycle: int  # 지배적 사이클 (봉 수)
    mean_reversion_force: float  # 평균회귀력
    trend_inertia: float  # 추세 관성


class PhysicsValidator:
    """
    물리학 기반 가격 신뢰도 검증기

    시장을 물리계로 모델링하여 가격 움직임의 신뢰성을 평가합니다.
    """

    def __init__(
        self,
        momentum_window: int = 14,
        energy_window: int = 20,
        entropy_window: int = 20,
        fft_window: int = 64
    ):
        self.momentum_window = momentum_window
        self.energy_window = energy_window
        self.entropy_window = entropy_window
        self.fft_window = fft_window

    def calculate_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        모멘텀 계산 (F = ma, 가격의 가속도)

        1차 미분: 속도 (가격 변화율)
        2차 미분: 가속도 (변화율의 변화율)
        """
        close = df['close']

        # 속도 (1차 미분): 가격 변화율
        velocity = close.pct_change()

        # 가속도 (2차 미분): 속도의 변화율
        acceleration = velocity.diff()

        # 모멘텀 = 가속도의 이동평균 (노이즈 제거)
        momentum = acceleration.rolling(self.momentum_window).mean()

        return momentum

    def calculate_energy(self, df: pd.DataFrame) -> pd.Series:
        """
        에너지 계산 (E = 거래량 × 변동폭)

        물리학의 에너지 보존 법칙 응용:
        - 거래량 = 질량
        - 가격 변동폭 = 속도
        - 에너지 = 1/2 × m × v²
        """
        volume = df['volume']
        price_range = df['high'] - df['low']

        # 정규화
        vol_norm = volume / volume.rolling(self.energy_window).mean()
        range_norm = price_range / price_range.rolling(self.energy_window).mean()

        # 에너지 = 거래량 × 변동폭²
        energy = vol_norm * (range_norm ** 2)

        return energy.rolling(5).mean()  # 스무딩

    def calculate_entropy(self, df: pd.DataFrame) -> pd.Series:
        """
        엔트로피 계산 (시장 혼란도/질서도)

        높은 엔트로피 = 무질서, 횡보, 불확실
        낮은 엔트로피 = 질서, 추세, 확실

        Shannon Entropy 사용
        """
        close = df['close']
        returns = close.pct_change().dropna()

        entropy_values = []

        for i in range(len(df)):
            if i < self.entropy_window:
                entropy_values.append(np.nan)
                continue

            window_returns = returns.iloc[max(0, i - self.entropy_window):i]

            if len(window_returns) < self.entropy_window // 2:
                entropy_values.append(np.nan)
                continue

            # 히스토그램으로 확률 분포 계산
            hist, _ = np.histogram(window_returns, bins=10, density=True)
            hist = hist[hist > 0]  # 0 제거

            # Shannon Entropy
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            entropy_values.append(entropy)

        return pd.Series(entropy_values, index=df.index)

    def analyze_waves_fft(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        FFT를 이용한 파동 분석

        가격의 주기적 패턴을 분석하여 지배적 사이클을 찾습니다.
        """
        close = df['close'].values[-self.fft_window:]

        if len(close) < self.fft_window:
            return {
                'dominant_cycle': 20,
                'wave_phase': 0.0,
                'wave_amplitude': 0.0,
                'cycles': []
            }

        # 디트렌딩 (선형 추세 제거)
        x = np.arange(len(close))
        coeffs = np.polyfit(x, close, 1)
        trend = np.polyval(coeffs, x)
        detrended = close - trend

        # FFT
        fft_values = fft(detrended)
        frequencies = fftfreq(len(detrended), 1)

        # 양수 주파수만
        positive_mask = frequencies > 0
        fft_magnitude = np.abs(fft_values[positive_mask])
        positive_freqs = frequencies[positive_mask]

        if len(fft_magnitude) == 0:
            return {
                'dominant_cycle': 20,
                'wave_phase': 0.0,
                'wave_amplitude': 0.0,
                'cycles': []
            }

        # 지배적 주파수 찾기
        dominant_idx = np.argmax(fft_magnitude)
        dominant_freq = positive_freqs[dominant_idx]

        # 주기 = 1/주파수
        dominant_cycle = int(1 / dominant_freq) if dominant_freq > 0 else 20

        # 위상 계산
        wave_phase = np.angle(fft_values[positive_mask][dominant_idx])

        # 진폭
        wave_amplitude = fft_magnitude[dominant_idx] / len(close)

        # 상위 3개 사이클
        top_indices = np.argsort(fft_magnitude)[-3:][::-1]
        cycles = []
        for idx in top_indices:
            freq = positive_freqs[idx]
            if freq > 0:
                cycles.append({
                    'period': int(1 / freq),
                    'amplitude': fft_magnitude[idx] / len(close),
                    'phase': np.angle(fft_values[positive_mask][idx])
                })

        return {
            'dominant_cycle': dominant_cycle,
            'wave_phase': wave_phase,
            'wave_amplitude': wave_amplitude,
            'cycles': cycles
        }

    def calculate_mean_reversion_force(self, df: pd.DataFrame, lookback: int = 50) -> pd.Series:
        """
        평균 회귀력 계산

        가격이 평균에서 멀어질수록 되돌아오려는 힘이 강해진다는 가정
        F = -k × (x - x_mean), 스프링 법칙 유사
        """
        close = df['close']
        sma = close.rolling(lookback).mean()

        # 평균으로부터의 편차 (정규화)
        std = close.rolling(lookback).std()
        deviation = (close - sma) / (std + 1e-10)

        # 회귀력 = -편차 (평균 쪽으로 당기는 힘)
        mean_reversion_force = -deviation

        return mean_reversion_force

    def calculate_trend_inertia(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """
        추세 관성 계산

        추세가 강할수록 지속되려는 관성이 크다
        관성 = 가격 변화의 일관성 (같은 방향 연속 횟수)
        """
        close = df['close']
        changes = close.diff()

        inertia_values = []

        for i in range(len(df)):
            if i < lookback:
                inertia_values.append(np.nan)
                continue

            window_changes = changes.iloc[i - lookback:i]

            # 양수/음수 비율
            positive_ratio = (window_changes > 0).sum() / lookback
            negative_ratio = (window_changes < 0).sum() / lookback

            # 관성 = 한 방향으로의 편향 정도
            inertia = abs(positive_ratio - negative_ratio) * 2  # 0~1 스케일

            # 방향 포함
            if positive_ratio > negative_ratio:
                inertia = inertia
            else:
                inertia = -inertia

            inertia_values.append(inertia)

        return pd.Series(inertia_values, index=df.index)

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        전체 물리학 분석 수행
        """
        # 각 지표 계산
        momentum = self.calculate_momentum(df)
        energy = self.calculate_energy(df)
        entropy = self.calculate_entropy(df)
        mean_reversion = self.calculate_mean_reversion_force(df)
        trend_inertia = self.calculate_trend_inertia(df)
        wave_analysis = self.analyze_waves_fft(df)

        # 최신 값
        current_momentum = float(momentum.iloc[-1]) if not np.isnan(momentum.iloc[-1]) else 0.0
        current_energy = float(energy.iloc[-1]) if not np.isnan(energy.iloc[-1]) else 1.0
        current_entropy = float(entropy.iloc[-1]) if not np.isnan(entropy.iloc[-1]) else 2.0
        current_reversion = float(mean_reversion.iloc[-1]) if not np.isnan(mean_reversion.iloc[-1]) else 0.0
        current_inertia = float(trend_inertia.iloc[-1]) if not np.isnan(trend_inertia.iloc[-1]) else 0.0

        metrics = PhysicsMetrics(
            momentum=current_momentum,
            energy=current_energy,
            entropy=current_entropy,
            wave_phase=wave_analysis['wave_phase'],
            wave_amplitude=wave_analysis['wave_amplitude'],
            dominant_cycle=wave_analysis['dominant_cycle'],
            mean_reversion_force=current_reversion,
            trend_inertia=current_inertia
        )

        return {
            'metrics': metrics,
            'momentum_series': momentum,
            'energy_series': energy,
            'entropy_series': entropy,
            'mean_reversion_series': mean_reversion,
            'trend_inertia_series': trend_inertia,
            'wave_analysis': wave_analysis
        }

    def validate_confluence_zone(
        self,
        df: pd.DataFrame,
        zone_price: float,
        zone_type: str  # 'support' or 'resistance'
    ) -> Tuple[float, Dict[str, Any]]:
        """
        컨플루언스 존의 물리학적 신뢰도 검증

        Args:
            df: OHLCV 데이터
            zone_price: 컨플루언스 존 가격
            zone_type: 'support' 또는 'resistance'

        Returns:
            (신뢰도 점수 0~1, 상세 분석)
        """
        analysis = self.analyze(df)
        metrics = analysis['metrics']

        score = 0.5  # 기본 점수
        details = {}

        current_price = float(df['close'].iloc[-1])
        distance_pct = abs(current_price - zone_price) / current_price * 100

        # 1. 모멘텀 분석
        if zone_type == 'support':
            # 지지선: 하락 모멘텀이 약해지면 좋음
            if metrics.momentum > -0.001:  # 가속도 감소/반전
                score += 0.1
                details['momentum'] = 'positive (deceleration)'
            else:
                score -= 0.05
                details['momentum'] = 'negative (acceleration down)'
        else:
            # 저항선: 상승 모멘텀이 약해지면 좋음
            if metrics.momentum < 0.001:
                score += 0.1
                details['momentum'] = 'positive (deceleration)'
            else:
                score -= 0.05
                details['momentum'] = 'negative (acceleration up)'

        # 2. 에너지 분석
        # 높은 에너지 = 변동성 높음 = 신뢰도 낮음 (돌파 가능성)
        if metrics.energy < 1.5:
            score += 0.1
            details['energy'] = f'low ({metrics.energy:.2f})'
        elif metrics.energy > 2.5:
            score -= 0.1
            details['energy'] = f'high ({metrics.energy:.2f})'
        else:
            details['energy'] = f'normal ({metrics.energy:.2f})'

        # 3. 엔트로피 분석
        # 낮은 엔트로피 = 질서 = 추세 지속 가능성
        if metrics.entropy < 2.5:
            score += 0.05
            details['entropy'] = f'ordered ({metrics.entropy:.2f})'
        elif metrics.entropy > 3.0:
            score -= 0.05
            details['entropy'] = f'chaotic ({metrics.entropy:.2f})'
        else:
            details['entropy'] = f'neutral ({metrics.entropy:.2f})'

        # 4. 평균 회귀력 분석
        if zone_type == 'support':
            # 지지선: 강한 하방 이탈 시 회귀력 높음 = 좋음
            if metrics.mean_reversion_force > 1.0:
                score += 0.15
                details['mean_reversion'] = f'strong buy force ({metrics.mean_reversion_force:.2f})'
            elif metrics.mean_reversion_force > 0.5:
                score += 0.08
                details['mean_reversion'] = f'moderate buy force ({metrics.mean_reversion_force:.2f})'
            else:
                details['mean_reversion'] = f'weak ({metrics.mean_reversion_force:.2f})'
        else:
            # 저항선: 강한 상방 이탈 시 회귀력 높음 = 좋음
            if metrics.mean_reversion_force < -1.0:
                score += 0.15
                details['mean_reversion'] = f'strong sell force ({metrics.mean_reversion_force:.2f})'
            elif metrics.mean_reversion_force < -0.5:
                score += 0.08
                details['mean_reversion'] = f'moderate sell force ({metrics.mean_reversion_force:.2f})'
            else:
                details['mean_reversion'] = f'weak ({metrics.mean_reversion_force:.2f})'

        # 5. 추세 관성 분석
        if zone_type == 'support':
            # 지지선: 하락 관성이 약해지면 좋음
            if metrics.trend_inertia > -0.3:
                score += 0.08
                details['trend_inertia'] = f'weakening downtrend ({metrics.trend_inertia:.2f})'
            else:
                score -= 0.05
                details['trend_inertia'] = f'strong downtrend ({metrics.trend_inertia:.2f})'
        else:
            # 저항선: 상승 관성이 약해지면 좋음
            if metrics.trend_inertia < 0.3:
                score += 0.08
                details['trend_inertia'] = f'weakening uptrend ({metrics.trend_inertia:.2f})'
            else:
                score -= 0.05
                details['trend_inertia'] = f'strong uptrend ({metrics.trend_inertia:.2f})'

        # 6. 파동 위상 분석
        # 사이클 저점 근처면 지지 신뢰도 높음
        phase = metrics.wave_phase
        phase_normalized = (phase + np.pi) / (2 * np.pi)  # 0~1 정규화

        if zone_type == 'support':
            # 사이클 저점(phase ≈ -π/2 or 3π/2) 근처면 좋음
            if 0.6 < phase_normalized < 0.9:
                score += 0.1
                details['wave_phase'] = f'near cycle bottom ({phase_normalized:.2f})'
            else:
                details['wave_phase'] = f'neutral ({phase_normalized:.2f})'
        else:
            # 사이클 고점(phase ≈ π/2) 근처면 좋음
            if 0.1 < phase_normalized < 0.4:
                score += 0.1
                details['wave_phase'] = f'near cycle top ({phase_normalized:.2f})'
            else:
                details['wave_phase'] = f'neutral ({phase_normalized:.2f})'

        # 점수 범위 제한
        score = max(0.0, min(1.0, score))

        details['dominant_cycle'] = f'{metrics.dominant_cycle} bars'
        details['final_score'] = score

        return score, details

    def calculate_arrival_probability(
        self,
        df: pd.DataFrame,
        target_price: float,
        max_bars: int = 100
    ) -> Dict[str, Any]:
        """
        목표 가격 도달 확률 계산

        물리학 지표를 기반으로 가격이 목표에 도달할 확률 추정
        """
        current_price = float(df['close'].iloc[-1])
        analysis = self.analyze(df)
        metrics = analysis['metrics']

        distance_pct = (target_price - current_price) / current_price * 100
        direction = 'up' if distance_pct > 0 else 'down'

        # 기본 확률 (거리 기반)
        # 거리가 멀수록 확률 낮음
        base_prob = max(0.1, 1.0 - abs(distance_pct) / 20)

        # 모멘텀 조정
        if direction == 'up' and metrics.momentum > 0:
            momentum_adj = 0.1
        elif direction == 'down' and metrics.momentum < 0:
            momentum_adj = 0.1
        else:
            momentum_adj = -0.1

        # 추세 관성 조정
        if direction == 'up' and metrics.trend_inertia > 0:
            inertia_adj = 0.15 * abs(metrics.trend_inertia)
        elif direction == 'down' and metrics.trend_inertia < 0:
            inertia_adj = 0.15 * abs(metrics.trend_inertia)
        else:
            inertia_adj = -0.1 * abs(metrics.trend_inertia)

        # 평균 회귀력 조정
        if direction == 'up' and metrics.mean_reversion_force > 0:
            reversion_adj = 0.1 * min(1, metrics.mean_reversion_force)
        elif direction == 'down' and metrics.mean_reversion_force < 0:
            reversion_adj = 0.1 * min(1, abs(metrics.mean_reversion_force))
        else:
            reversion_adj = -0.05

        # 에너지 조정 (높은 에너지 = 큰 움직임 가능)
        if metrics.energy > 1.5:
            energy_adj = 0.1
        else:
            energy_adj = 0

        # 최종 확률
        probability = base_prob + momentum_adj + inertia_adj + reversion_adj + energy_adj
        probability = max(0.05, min(0.95, probability))

        # 예상 소요 시간 (봉 수)
        avg_move = df['close'].pct_change().abs().rolling(20).mean().iloc[-1]
        if avg_move > 0:
            estimated_bars = int(abs(distance_pct) / (avg_move * 100))
        else:
            estimated_bars = max_bars

        estimated_bars = min(estimated_bars, max_bars)

        return {
            'target_price': target_price,
            'current_price': current_price,
            'distance_pct': distance_pct,
            'direction': direction,
            'probability': probability,
            'estimated_bars': estimated_bars,
            'adjustments': {
                'base': base_prob,
                'momentum': momentum_adj,
                'inertia': inertia_adj,
                'reversion': reversion_adj,
                'energy': energy_adj
            },
            'metrics': {
                'momentum': metrics.momentum,
                'energy': metrics.energy,
                'trend_inertia': metrics.trend_inertia,
                'mean_reversion_force': metrics.mean_reversion_force
            }
        }
