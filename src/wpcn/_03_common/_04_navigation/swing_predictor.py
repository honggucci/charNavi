"""
스윙 포인트 확률 예측기
Swing Point Probability Predictor

RSI 다이버전스 + 피보나치 + 물리학 모델을 결합하여
지그재그 스윙 포인트 도달 확률을 예측합니다.

핵심 공식:
P(스윙 → Fib레벨) = P(RSI반전) × P(가격도달|변동성) × P(시간도달|FFT주기)
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import norm


@dataclass
class SwingPrediction:
    """스윙 포인트 예측 결과"""
    target_price: float           # 타겟 가격
    fib_level: float             # 피보나치 레벨 (0.382, 0.5, 0.618)
    probability: float           # 도달 확률 (0~1)
    expected_bars: int           # 예상 도달 봉 수
    confidence: float            # 신뢰도 (0~1)
    direction: str               # 'up' or 'down'
    components: Dict[str, float] # 확률 구성 요소


@dataclass
class PriceBoundary:
    """RSI 다이버전스 기반 가격 바운더리"""
    upper: float                 # 상한선
    lower: float                 # 하한선
    center: float                # 중심선
    divergence_type: str         # 'bullish' or 'bearish' or 'none'
    divergence_strength: float   # 다이버전스 강도 (0~1)


class SwingPredictor:
    """
    스윙 포인트 확률 예측기

    RSI 다이버전스로 가격 바운더리를 구하고,
    물리학 모델로 피보나치 레벨 도달 확률을 계산합니다.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        divergence_lookback: int = 30,
        fft_window: int = 64,
        volatility_window: int = 20
    ):
        self.rsi_period = rsi_period
        self.divergence_lookback = divergence_lookback
        self.fft_window = fft_window
        self.volatility_window = volatility_window

        # 피보나치 되돌림 레벨
        self.fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]

    # =========================================================================
    # 1. RSI 다이버전스 기반 가격 바운더리 계산
    # =========================================================================

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def find_divergences(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        RSI 다이버전스 감지

        Bullish Divergence: 가격 Lower Low + RSI Higher Low
        Bearish Divergence: 가격 Higher High + RSI Lower High
        """
        close = df['close']
        rsi = self.calculate_rsi(close)

        divergences = []

        # 지그재그 피벗 찾기 (간단한 버전)
        window = 5
        price_highs = []
        price_lows = []
        rsi_highs = []
        rsi_lows = []

        for i in range(window, len(df) - window):
            # Local High
            if close.iloc[i] == close.iloc[i-window:i+window+1].max():
                price_highs.append((i, close.iloc[i], rsi.iloc[i]))

            # Local Low
            if close.iloc[i] == close.iloc[i-window:i+window+1].min():
                price_lows.append((i, close.iloc[i], rsi.iloc[i]))

        # Bullish Divergence 감지 (최근 2개 저점 비교)
        if len(price_lows) >= 2:
            prev_low = price_lows[-2]
            curr_low = price_lows[-1]

            # 가격: Lower Low, RSI: Higher Low
            if curr_low[1] < prev_low[1] and curr_low[2] > prev_low[2]:
                strength = (curr_low[2] - prev_low[2]) / 30  # RSI 차이로 강도 계산
                divergences.append({
                    'type': 'bullish',
                    'price_idx': curr_low[0],
                    'price': curr_low[1],
                    'rsi': curr_low[2],
                    'strength': min(1.0, abs(strength))
                })

        # Bearish Divergence 감지 (최근 2개 고점 비교)
        if len(price_highs) >= 2:
            prev_high = price_highs[-2]
            curr_high = price_highs[-1]

            # 가격: Higher High, RSI: Lower High
            if curr_high[1] > prev_high[1] and curr_high[2] < prev_high[2]:
                strength = (prev_high[2] - curr_high[2]) / 30
                divergences.append({
                    'type': 'bearish',
                    'price_idx': curr_high[0],
                    'price': curr_high[1],
                    'rsi': curr_high[2],
                    'strength': min(1.0, abs(strength))
                })

        return divergences

    def calculate_price_boundary(self, df: pd.DataFrame) -> PriceBoundary:
        """
        RSI 다이버전스 기반 가격 바운더리 계산

        바운더리 = 최근 고점/저점 + 다이버전스 강도 보정
        """
        lookback = min(self.divergence_lookback, len(df))
        recent_df = df.iloc[-lookback:]

        high = recent_df['high'].max()
        low = recent_df['low'].min()
        current = df['close'].iloc[-1]

        # 다이버전스 감지
        divergences = self.find_divergences(df.iloc[-100:])

        div_type = 'none'
        div_strength = 0.0

        if divergences:
            latest_div = divergences[-1]
            div_type = latest_div['type']
            div_strength = latest_div['strength']

        # ATR 기반 바운더리 확장
        atr = self._calculate_atr(df, 14)
        atr_mult = 1.5 + div_strength  # 다이버전스가 강할수록 확장

        # 바운더리 계산
        if div_type == 'bullish':
            # Bullish: 상방 확장
            upper = high + atr * atr_mult
            lower = low
        elif div_type == 'bearish':
            # Bearish: 하방 확장
            upper = high
            lower = low - atr * atr_mult
        else:
            # 중립
            upper = high + atr
            lower = low - atr

        center = (upper + lower) / 2

        return PriceBoundary(
            upper=upper,
            lower=lower,
            center=center,
            divergence_type=div_type,
            divergence_strength=div_strength
        )

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        """ATR 계산"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]

        return float(atr)

    # =========================================================================
    # 2. 피보나치 타겟 레벨 계산
    # =========================================================================

    def calculate_fib_targets(
        self,
        boundary: PriceBoundary,
        current_price: float
    ) -> List[Dict[str, Any]]:
        """
        바운더리 내 피보나치 타겟 레벨 계산
        """
        targets = []

        # 상승 타겟 (현재가 → 상한선)
        if current_price < boundary.upper:
            range_up = boundary.upper - current_price

            for fib in self.fib_levels:
                target_price = current_price + range_up * fib
                if target_price <= boundary.upper:
                    targets.append({
                        'price': target_price,
                        'fib_level': fib,
                        'direction': 'up',
                        'distance_pct': (target_price - current_price) / current_price * 100
                    })

        # 하락 타겟 (현재가 → 하한선)
        if current_price > boundary.lower:
            range_down = current_price - boundary.lower

            for fib in self.fib_levels:
                target_price = current_price - range_down * fib
                if target_price >= boundary.lower:
                    targets.append({
                        'price': target_price,
                        'fib_level': fib,
                        'direction': 'down',
                        'distance_pct': (target_price - current_price) / current_price * 100
                    })

        return targets

    # =========================================================================
    # 3. 물리학 기반 확률 계산
    # =========================================================================

    def _analyze_fft_cycle(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        FFT 사이클 분석

        Returns:
            dominant_cycle: 지배적 주기 (봉 수)
            current_phase: 현재 위상 (0~1, 0=저점, 0.5=고점)
            amplitude: 진폭
        """
        close = df['close'].values[-self.fft_window:]

        if len(close) < self.fft_window:
            return {'dominant_cycle': 20, 'current_phase': 0.5, 'amplitude': 0}

        # 디트렌딩
        x = np.arange(len(close))
        coeffs = np.polyfit(x, close, 1)
        trend = np.polyval(coeffs, x)
        detrended = close - trend

        # FFT
        fft_values = fft(detrended)
        frequencies = fftfreq(len(detrended), 1)

        positive_mask = frequencies > 0
        fft_magnitude = np.abs(fft_values[positive_mask])
        positive_freqs = frequencies[positive_mask]

        if len(fft_magnitude) == 0:
            return {'dominant_cycle': 20, 'current_phase': 0.5, 'amplitude': 0}

        # 지배적 주파수
        dominant_idx = np.argmax(fft_magnitude)
        dominant_freq = positive_freqs[dominant_idx]

        dominant_cycle = int(1 / dominant_freq) if dominant_freq > 0 else 20
        dominant_cycle = max(5, min(100, dominant_cycle))  # 5~100봉 범위 제한

        # 현재 위상 (힐베르트 변환 대용)
        phase = np.angle(fft_values[positive_mask][dominant_idx])
        current_phase = (phase + np.pi) / (2 * np.pi)  # 0~1 정규화

        # 진폭
        amplitude = fft_magnitude[dominant_idx] / len(close)

        return {
            'dominant_cycle': dominant_cycle,
            'current_phase': current_phase,
            'amplitude': amplitude
        }

    def _black_scholes_probability(
        self,
        current_price: float,
        target_price: float,
        volatility: float,
        time_bars: int
    ) -> float:
        """
        블랙숄즈 모형 기반 도달 확률

        P(S_T > K) = N(d2) for call-like probability
        """
        if volatility <= 0 or time_bars <= 0:
            return 0.5

        # 시간 정규화 (1년 = 35040봉, 15분봉 기준)
        T = time_bars / 35040

        # d2 계산 (무위험 이자율 = 0 가정)
        log_ratio = np.log(target_price / current_price)
        d2 = (log_ratio - 0.5 * volatility**2 * T) / (volatility * np.sqrt(T) + 1e-10)

        # 방향에 따른 확률
        if target_price > current_price:
            # 상승 확률
            prob = norm.cdf(-d2)  # P(S > K)
        else:
            # 하락 확률
            prob = norm.cdf(d2)  # P(S < K)

        return float(prob)

    def _boltzmann_distribution(
        self,
        current_price: float,
        target_price: float,
        temperature: float  # 시장 변동성
    ) -> float:
        """
        볼츠만 분포 기반 가격 전이 확률

        P(E) ∝ exp(-E/kT)
        E = |target - current| (에너지 = 가격 거리)
        T = 변동성 (온도)
        """
        if temperature <= 0:
            temperature = 0.01

        # 에너지 = 정규화된 가격 거리
        energy = abs(target_price - current_price) / current_price

        # 볼츠만 확률
        prob = np.exp(-energy / temperature)

        return float(prob)

    def _calculate_time_probability(
        self,
        fft_result: Dict[str, Any],
        target_direction: str
    ) -> Tuple[float, int]:
        """
        FFT 사이클 기반 시간 확률 계산

        Returns:
            probability: 현재 타이밍이 적절한지 확률
            expected_bars: 예상 도달 봉 수
        """
        phase = fft_result['current_phase']
        cycle = fft_result['dominant_cycle']

        if target_direction == 'up':
            # 상승: 사이클 저점(phase ≈ 0.75)에서 좋음
            # 사이클 고점(phase ≈ 0.25)에서 나쁨
            if 0.6 < phase < 0.9:
                time_prob = 0.8
                bars_to_target = int(cycle * (1 - phase))
            elif phase < 0.3:
                time_prob = 0.3
                bars_to_target = int(cycle * (0.75 - phase))
            else:
                time_prob = 0.5
                bars_to_target = int(cycle * 0.5)

        else:  # down
            # 하락: 사이클 고점(phase ≈ 0.25)에서 좋음
            if 0.1 < phase < 0.4:
                time_prob = 0.8
                bars_to_target = int(cycle * phase)
            elif phase > 0.7:
                time_prob = 0.3
                bars_to_target = int(cycle * (phase - 0.25))
            else:
                time_prob = 0.5
                bars_to_target = int(cycle * 0.5)

        return time_prob, max(1, bars_to_target)

    # =========================================================================
    # 4. 종합 스윙 예측
    # =========================================================================

    def predict_swing(
        self,
        df: pd.DataFrame,
        target_fib_levels: List[float] = None
    ) -> List[SwingPrediction]:
        """
        스윙 포인트 확률 예측

        Args:
            df: OHLCV 데이터
            target_fib_levels: 타겟 피보나치 레벨 (기본: [0.382, 0.5, 0.618])

        Returns:
            List[SwingPrediction]: 각 피보나치 레벨별 예측
        """
        if target_fib_levels is None:
            target_fib_levels = [0.382, 0.5, 0.618]

        current_price = float(df['close'].iloc[-1])

        # 1. RSI 다이버전스 기반 가격 바운더리
        boundary = self.calculate_price_boundary(df)

        # 2. 피보나치 타겟 레벨
        fib_targets = self.calculate_fib_targets(boundary, current_price)

        # 3. FFT 사이클 분석
        fft_result = self._analyze_fft_cycle(df)

        # 4. 변동성 계산
        returns = df['close'].pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(35040))  # 연간화

        predictions = []

        for target in fib_targets:
            if target['fib_level'] not in target_fib_levels:
                continue

            target_price = target['price']
            direction = target['direction']

            # 4-1. RSI 다이버전스 확률
            if boundary.divergence_type == 'bullish' and direction == 'up':
                rsi_prob = 0.6 + boundary.divergence_strength * 0.3
            elif boundary.divergence_type == 'bearish' and direction == 'down':
                rsi_prob = 0.6 + boundary.divergence_strength * 0.3
            elif boundary.divergence_type == 'none':
                rsi_prob = 0.5
            else:
                rsi_prob = 0.3  # 역방향

            # 4-2. 블랙숄즈 확률 (변동성 기반)
            bs_time = fft_result['dominant_cycle']
            bs_prob = self._black_scholes_probability(
                current_price, target_price, volatility, bs_time
            )

            # 4-3. 볼츠만 확률 (가격 거리 기반)
            temperature = volatility * 0.1  # 변동성을 온도로
            boltz_prob = self._boltzmann_distribution(
                current_price, target_price, temperature
            )

            # 4-4. FFT 시간 확률
            time_prob, expected_bars = self._calculate_time_probability(
                fft_result, direction
            )

            # 5. 종합 확률 계산
            # P = P(RSI) × P(BS|vol) × P(Boltz|dist) × P(Time|FFT)
            combined_prob = (
                rsi_prob * 0.30 +      # RSI 다이버전스 30%
                bs_prob * 0.25 +       # 블랙숄즈 25%
                boltz_prob * 0.20 +    # 볼츠만 20%
                time_prob * 0.25       # FFT 타이밍 25%
            )

            # 거리 페널티 (먼 타겟일수록 확률 감소)
            distance = abs(target['distance_pct'])
            distance_penalty = max(0.5, 1.0 - distance / 20)
            combined_prob *= distance_penalty

            # 신뢰도 = 구성 요소들의 일관성
            probs = [rsi_prob, bs_prob, boltz_prob, time_prob]
            confidence = 1.0 - np.std(probs)  # 분산이 작을수록 신뢰도 높음

            predictions.append(SwingPrediction(
                target_price=target_price,
                fib_level=target['fib_level'],
                probability=min(0.95, max(0.05, combined_prob)),
                expected_bars=expected_bars,
                confidence=confidence,
                direction=direction,
                components={
                    'rsi_divergence': rsi_prob,
                    'black_scholes': bs_prob,
                    'boltzmann': boltz_prob,
                    'fft_timing': time_prob,
                    'distance_penalty': distance_penalty
                }
            ))

        # 확률 순으로 정렬
        predictions.sort(key=lambda x: x.probability, reverse=True)

        return predictions

    def get_best_entry(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        최적의 진입 포인트 추천

        Returns:
            {
                'direction': 'long' or 'short',
                'probability': 도달 확률,
                'target': 타겟 가격,
                'stop_loss': 손절 가격,
                'take_profit': 익절 가격,
                'expected_bars': 예상 소요 봉 수
            }
        """
        predictions = self.predict_swing(df)

        if not predictions:
            return None

        # 가장 확률 높은 예측
        best = predictions[0]

        # 최소 확률 필터 (0.50 이상이면 방향성 있음)
        if best.probability < 0.50:
            return None

        current_price = float(df['close'].iloc[-1])

        if best.direction == 'up':
            return {
                'direction': 'long',
                'probability': best.probability,
                'confidence': best.confidence,
                'target': best.target_price,
                'fib_level': best.fib_level,
                'stop_loss': current_price * 0.98,  # -2%
                'take_profit': best.target_price,
                'expected_bars': best.expected_bars,
                'components': best.components
            }
        else:
            return {
                'direction': 'short',
                'probability': best.probability,
                'confidence': best.confidence,
                'target': best.target_price,
                'fib_level': best.fib_level,
                'stop_loss': current_price * 1.02,  # +2%
                'take_profit': best.target_price,
                'expected_bars': best.expected_bars,
                'components': best.components
            }

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        전체 분석 리포트 생성
        """
        boundary = self.calculate_price_boundary(df)
        predictions = self.predict_swing(df)
        best_entry = self.get_best_entry(df)
        fft_result = self._analyze_fft_cycle(df)

        current_price = float(df['close'].iloc[-1])

        # 종합 방향성
        up_probs = [p.probability for p in predictions if p.direction == 'up']
        down_probs = [p.probability for p in predictions if p.direction == 'down']

        avg_up = np.mean(up_probs) if up_probs else 0
        avg_down = np.mean(down_probs) if down_probs else 0

        if avg_up > avg_down + 0.1:
            overall_bias = 'bullish'
        elif avg_down > avg_up + 0.1:
            overall_bias = 'bearish'
        else:
            overall_bias = 'neutral'

        return {
            'current_price': current_price,
            'boundary': {
                'upper': boundary.upper,
                'lower': boundary.lower,
                'center': boundary.center,
                'divergence': boundary.divergence_type,
                'divergence_strength': boundary.divergence_strength
            },
            'cycle': {
                'dominant_period': fft_result['dominant_cycle'],
                'current_phase': fft_result['current_phase'],
                'amplitude': fft_result['amplitude']
            },
            'predictions': [
                {
                    'target': p.target_price,
                    'fib_level': p.fib_level,
                    'direction': p.direction,
                    'probability': p.probability,
                    'confidence': p.confidence,
                    'expected_bars': p.expected_bars
                }
                for p in predictions[:5]  # 상위 5개
            ],
            'best_entry': best_entry,
            'overall_bias': overall_bias,
            'avg_up_probability': avg_up,
            'avg_down_probability': avg_down
        }


# 테스트 코드
if __name__ == "__main__":
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(REPO_ROOT / 'src'))

    from wpcn.data.loaders import load_parquet

    # 데이터 로드
    data_path = REPO_ROOT / 'data' / 'raw' / 'binance_BTC-USDT_15m_90d.parquet'
    if data_path.exists():
        df = load_parquet(str(data_path))
        print(f"데이터 로드: {len(df)} 캔들")

        # 예측기 생성
        predictor = SwingPredictor()

        # 분석
        analysis = predictor.analyze(df)

        print("\n" + "="*60)
        print("스윙 포인트 확률 예측 분석")
        print("="*60)

        print(f"\n현재가: ${analysis['current_price']:,.2f}")
        print(f"바운더리: ${analysis['boundary']['lower']:,.2f} ~ ${analysis['boundary']['upper']:,.2f}")
        print(f"다이버전스: {analysis['boundary']['divergence']} (강도: {analysis['boundary']['divergence_strength']:.2f})")

        print(f"\n사이클: {analysis['cycle']['dominant_period']}봉")
        print(f"현재 위상: {analysis['cycle']['current_phase']:.2f} (0=저점, 0.5=고점)")

        print(f"\n전체 편향: {analysis['overall_bias'].upper()}")
        print(f"상승 평균 확률: {analysis['avg_up_probability']:.1%}")
        print(f"하락 평균 확률: {analysis['avg_down_probability']:.1%}")

        print("\n--- 상위 5개 예측 ---")
        for pred in analysis['predictions']:
            print(f"  {pred['direction'].upper()}: ${pred['target']:,.2f} "
                  f"(Fib {pred['fib_level']}) - "
                  f"확률 {pred['probability']:.1%}, "
                  f"{pred['expected_bars']}봉 예상")

        if analysis['best_entry']:
            entry = analysis['best_entry']
            print("\n--- 최적 진입점 ---")
            print(f"  방향: {entry['direction'].upper()}")
            print(f"  확률: {entry['probability']:.1%}")
            print(f"  타겟: ${entry['target']:,.2f} (Fib {entry['fib_level']})")
            print(f"  손절: ${entry['stop_loss']:,.2f}")
            print(f"  예상 봉수: {entry['expected_bars']}")
            print("\n  구성 요소:")
            for k, v in entry['components'].items():
                print(f"    {k}: {v:.2%}")
        else:
            print("\n[!] 현재 적합한 진입점 없음")
    else:
        print(f"데이터 파일이 없습니다: {data_path}")
