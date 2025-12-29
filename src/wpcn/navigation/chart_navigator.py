"""
차트 네비게이터 - 통합 분석 시스템
Chart Navigator - Integrated Analysis System

확률론적 와이코프 스나이퍼 매매를 위한 핵심 네비게이션 시스템
모든 분석 모듈을 통합하여 최적의 진입/청산 포인트를 제시합니다.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .fibonacci_confluence import MTFFibonacciConfluence, ConfluenceZone
from .physics_validator import PhysicsValidator
from .probability_calculator import ProbabilityCalculator, calculate_rsi, calculate_stoch_rsi


@dataclass
class WyckoffPhase:
    """와이코프 페이즈 정보"""
    structure: str  # 'accumulation', 'distribution', 're_accumulation', 're_distribution', 'markup', 'markdown'
    phase: str  # 'A', 'B', 'C', 'D', 'E'
    progress: float  # 0~1 (페이즈 진행도)
    confidence: float  # 0~1 (판단 신뢰도)
    description: str


@dataclass
class NavigationReport:
    """네비게이션 리포트"""
    timestamp: Any
    current_price: float

    # 와이코프 분석
    wyckoff_phase: WyckoffPhase

    # SC/BC 예측
    sc_prediction: Optional[Dict]
    bc_prediction: Optional[Dict]

    # 컨플루언스 존
    nearest_support: Optional[ConfluenceZone]
    nearest_resistance: Optional[ConfluenceZone]
    all_confluence_zones: List[ConfluenceZone]

    # 물리학 지표
    physics_metrics: Dict

    # 모멘텀 지표
    momentum_metrics: Dict

    # 방향 확률
    direction_probabilities: Dict[str, float]

    # 추천 액션
    recommended_action: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size_pct: float

    # 신뢰도
    overall_confidence: float


class ChartNavigator:
    """
    차트 네비게이터 - 통합 분석 시스템

    1. MTF 피보나치 컨플루언스로 핵심 가격대 식별
    2. 물리학 검증으로 가격대 신뢰도 평가
    3. RSI 다이버전스 + Stoch RSI로 타이밍 분석
    4. 와이코프 페이즈 판정으로 현재 위치 파악
    5. 종합 확률 계산으로 최적 진입/청산 제시
    """

    def __init__(
        self,
        timeframes: List[str] = None,
        confluence_tolerance: float = 0.3,
        min_confluence: int = 2
    ):
        self.timeframes = timeframes or ['15m', '1h', '4h', '1d']

        # 서브 모듈 초기화
        self.fib_analyzer = MTFFibonacciConfluence(
            timeframes=self.timeframes,
            confluence_tolerance=confluence_tolerance,
            min_confluence=min_confluence
        )
        self.physics_validator = PhysicsValidator()
        self.prob_calculator = ProbabilityCalculator()

    def detect_wyckoff_phase(
        self,
        df: pd.DataFrame,
        confluence_analysis: Dict,
        physics_analysis: Dict,
        momentum_analysis: Dict
    ) -> WyckoffPhase:
        """
        와이코프 페이즈 종합 판정

        가격 구조 + 거래량 + 물리학 지표를 종합하여 현재 위치 판단
        """
        close = df['close']
        volume = df['volume']
        current_price = float(close.iloc[-1])

        # 최근 가격 범위 분석
        lookback = min(200, len(df))
        recent_high = float(df['high'].iloc[-lookback:].max())
        recent_low = float(df['low'].iloc[-lookback:].min())
        price_range = recent_high - recent_low

        if price_range == 0:
            price_range = current_price * 0.01

        # 현재 가격 위치 (0=저점, 1=고점)
        price_position = (current_price - recent_low) / price_range

        # 거래량 트렌드
        vol_sma_short = volume.rolling(10).mean()
        vol_sma_long = volume.rolling(50).mean()
        volume_trend = 'increasing' if vol_sma_short.iloc[-1] > vol_sma_long.iloc[-1] else 'decreasing'

        # 물리학 지표
        physics_metrics = physics_analysis['metrics']
        momentum = physics_metrics.momentum
        energy = physics_metrics.energy
        mean_reversion = physics_metrics.mean_reversion_force

        # 모멘텀 지표
        stoch_zone = momentum_analysis['stoch_zone']
        recent_div = momentum_analysis['recent_divergence']

        # 구조 판정
        structure = 'unknown'
        phase = 'B'
        progress = 0.5
        confidence = 0.5

        # === 매집 (Accumulation) 판정 ===
        if price_position < 0.3:  # 저점 근처
            if stoch_zone == 'oversold':
                structure = 'accumulation'

                if recent_div and recent_div.divergence_type == 'bullish':
                    # Bullish 다이버전스 = Phase C (Spring) 가능성
                    phase = 'C'
                    progress = 0.7
                    confidence = 0.7 + recent_div.strength * 0.2
                elif energy < 1.0 and volume_trend == 'decreasing':
                    # 낮은 에너지 + 거래량 감소 = Phase B
                    phase = 'B'
                    progress = 0.5
                    confidence = 0.6
                elif energy > 1.5 and mean_reversion > 0.5:
                    # 높은 에너지 + 강한 회귀력 = Phase A (SC 형성)
                    phase = 'A'
                    progress = 0.3
                    confidence = 0.65
                else:
                    phase = 'B'
                    progress = 0.4
                    confidence = 0.5

        # === 분산 (Distribution) 판정 ===
        elif price_position > 0.7:  # 고점 근처
            if stoch_zone == 'overbought':
                structure = 'distribution'

                if recent_div and recent_div.divergence_type == 'bearish':
                    # Bearish 다이버전스 = Phase C (UTAD) 가능성
                    phase = 'C'
                    progress = 0.7
                    confidence = 0.7 + recent_div.strength * 0.2
                elif energy < 1.0 and volume_trend == 'decreasing':
                    phase = 'B'
                    progress = 0.5
                    confidence = 0.6
                elif energy > 1.5 and mean_reversion < -0.5:
                    phase = 'A'
                    progress = 0.3
                    confidence = 0.65
                else:
                    phase = 'B'
                    progress = 0.4
                    confidence = 0.5

        # === 추세 중 재매집/재분산 판정 ===
        elif 0.3 <= price_position <= 0.7:
            trend_inertia = physics_metrics.trend_inertia

            if trend_inertia > 0.3:  # 상승 추세
                if stoch_zone == 'oversold':
                    structure = 're_accumulation'
                    phase = 'B'
                    progress = 0.6
                    confidence = 0.6
                else:
                    structure = 'markup'
                    phase = 'D'
                    progress = 0.5
                    confidence = 0.55
            elif trend_inertia < -0.3:  # 하락 추세
                if stoch_zone == 'overbought':
                    structure = 're_distribution'
                    phase = 'B'
                    progress = 0.6
                    confidence = 0.6
                else:
                    structure = 'markdown'
                    phase = 'D'
                    progress = 0.5
                    confidence = 0.55
            else:
                # 횡보
                structure = 'accumulation' if mean_reversion > 0 else 'distribution'
                phase = 'B'
                progress = 0.5
                confidence = 0.4

        # 설명 생성
        description = self._generate_phase_description(structure, phase, progress)

        return WyckoffPhase(
            structure=structure,
            phase=phase,
            progress=progress,
            confidence=confidence,
            description=description
        )

    def _generate_phase_description(
        self,
        structure: str,
        phase: str,
        progress: float
    ) -> str:
        """페이즈 설명 생성"""
        structure_desc = {
            'accumulation': '매집 구간',
            'distribution': '분산 구간',
            're_accumulation': '상승 중 재매집',
            're_distribution': '하락 중 재분산',
            'markup': '상승 추세',
            'markdown': '하락 추세',
            'unknown': '불확실'
        }

        phase_desc = {
            'A': 'Phase A (SC/BC 형성 중)',
            'B': 'Phase B (횡보/테스트)',
            'C': 'Phase C (Spring/UTAD 임박)',
            'D': 'Phase D (추세 진행)',
            'E': 'Phase E (추세 가속)'
        }

        progress_str = f"{progress*100:.0f}%"

        return f"{structure_desc.get(structure, structure)} {phase_desc.get(phase, phase)} (진행도: {progress_str})"

    def generate_recommendation(
        self,
        wyckoff_phase: WyckoffPhase,
        sc_prediction: Optional[Dict],
        bc_prediction: Optional[Dict],
        direction_probs: Dict[str, float],
        momentum_analysis: Dict,
        current_price: float
    ) -> Tuple[str, Optional[float], Optional[float], Optional[float], float]:
        """
        매매 추천 생성

        Returns:
            (action, entry_price, stop_loss, take_profit, position_size_pct)
        """
        action = 'WAIT'
        entry_price = None
        stop_loss = None
        take_profit = None
        position_size_pct = 0.0

        structure = wyckoff_phase.structure
        phase = wyckoff_phase.phase
        confidence = wyckoff_phase.confidence

        stoch_zone = momentum_analysis['stoch_zone']
        recent_div = momentum_analysis['recent_divergence']

        # === 매집 구간 롱 전략 ===
        if structure in ['accumulation', 're_accumulation']:
            if phase == 'C' and sc_prediction and confidence > 0.6:
                # Phase C + SC 예측 = 롱 진입 준비
                action = 'LONG_READY'
                entry_price = sc_prediction['price']
                stop_loss = entry_price * 0.985  # -1.5%
                take_profit = entry_price * 1.03  # +3%
                position_size_pct = min(0.2, confidence * 0.3)

                if stoch_zone == 'oversold' and recent_div and recent_div.divergence_type == 'bullish':
                    action = 'LONG_NOW'
                    entry_price = current_price
                    position_size_pct = min(0.25, confidence * 0.35)

            elif phase == 'B' and sc_prediction:
                action = 'ACCUMULATE'
                entry_price = sc_prediction['price']
                stop_loss = entry_price * 0.98
                take_profit = entry_price * 1.025
                position_size_pct = 0.03  # 3% 분할 매수

        # === 분산 구간 숏 전략 ===
        elif structure in ['distribution', 're_distribution']:
            if phase == 'C' and bc_prediction and confidence > 0.6:
                action = 'SHORT_READY'
                entry_price = bc_prediction['price']
                stop_loss = entry_price * 1.015
                take_profit = entry_price * 0.97
                position_size_pct = min(0.2, confidence * 0.3)

                if stoch_zone == 'overbought' and recent_div and recent_div.divergence_type == 'bearish':
                    action = 'SHORT_NOW'
                    entry_price = current_price
                    position_size_pct = min(0.25, confidence * 0.35)

            elif phase == 'B' and bc_prediction:
                action = 'DISTRIBUTE'
                entry_price = bc_prediction['price']
                stop_loss = entry_price * 1.02
                take_profit = entry_price * 0.975
                position_size_pct = 0.03

        # === 추세 추종 ===
        elif structure == 'markup':
            if direction_probs['up'] > 0.6:
                action = 'TREND_LONG'
                entry_price = current_price
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.03
                position_size_pct = 0.1

        elif structure == 'markdown':
            if direction_probs['down'] > 0.6:
                action = 'TREND_SHORT'
                entry_price = current_price
                stop_loss = current_price * 1.02
                take_profit = current_price * 0.97
                position_size_pct = 0.1

        return action, entry_price, stop_loss, take_profit, position_size_pct

    def analyze(self, df: pd.DataFrame) -> NavigationReport:
        """
        전체 차트 네비게이션 분석 수행
        """
        current_price = float(df['close'].iloc[-1])
        timestamp = df.index[-1]

        # 1. 피보나치 컨플루언스 분석
        confluence_analysis = self.fib_analyzer.analyze(df, current_price)
        sc_bc_prediction = self.fib_analyzer.get_sc_bc_prediction(confluence_analysis)

        # 2. 물리학 분석
        physics_analysis = self.physics_validator.analyze(df)

        # 3. 모멘텀 분석
        momentum_analysis = self.prob_calculator.analyze_momentum_indicators(df)

        # 4. 와이코프 페이즈 판정
        wyckoff_phase = self.detect_wyckoff_phase(
            df,
            confluence_analysis,
            physics_analysis,
            momentum_analysis
        )

        # 5. 방향 확률 계산
        physics_metrics_dict = {
            'momentum': physics_analysis['metrics'].momentum,
            'trend_inertia': physics_analysis['metrics'].trend_inertia
        }
        direction_probs = self.prob_calculator.calculate_directional_probability(
            df,
            momentum_analysis,
            physics_metrics_dict
        )

        # 6. 컨플루언스 존 물리학 검증
        sc_prediction = sc_bc_prediction.get('sc_prediction')
        bc_prediction = sc_bc_prediction.get('bc_prediction')

        if sc_prediction:
            physics_score, physics_details = self.physics_validator.validate_confluence_zone(
                df, sc_prediction['price'], 'support'
            )
            sc_prediction['physics_confidence'] = physics_score
            sc_prediction['physics_details'] = physics_details

            # 도달 확률 계산
            arrival = self.physics_validator.calculate_arrival_probability(
                df, sc_prediction['price']
            )
            sc_prediction['arrival_probability'] = arrival['probability']
            sc_prediction['estimated_bars'] = arrival['estimated_bars']

        if bc_prediction:
            physics_score, physics_details = self.physics_validator.validate_confluence_zone(
                df, bc_prediction['price'], 'resistance'
            )
            bc_prediction['physics_confidence'] = physics_score
            bc_prediction['physics_details'] = physics_details

            arrival = self.physics_validator.calculate_arrival_probability(
                df, bc_prediction['price']
            )
            bc_prediction['arrival_probability'] = arrival['probability']
            bc_prediction['estimated_bars'] = arrival['estimated_bars']

        # 7. 매매 추천 생성
        action, entry, sl, tp, size = self.generate_recommendation(
            wyckoff_phase,
            sc_prediction,
            bc_prediction,
            direction_probs,
            momentum_analysis,
            current_price
        )

        # 8. 전체 신뢰도 계산
        overall_confidence = wyckoff_phase.confidence

        if sc_prediction:
            overall_confidence = (overall_confidence + sc_prediction.get('physics_confidence', 0.5)) / 2
        if bc_prediction:
            overall_confidence = (overall_confidence + bc_prediction.get('physics_confidence', 0.5)) / 2

        # 물리학 메트릭 정리
        pm = physics_analysis['metrics']
        physics_metrics = {
            'momentum': pm.momentum,
            'energy': pm.energy,
            'entropy': pm.entropy,
            'mean_reversion_force': pm.mean_reversion_force,
            'trend_inertia': pm.trend_inertia,
            'dominant_cycle': pm.dominant_cycle,
            'wave_phase': pm.wave_phase,
            'wave_amplitude': pm.wave_amplitude
        }

        # 모멘텀 메트릭 정리
        momentum_metrics = {
            'rsi': momentum_analysis['current_rsi'],
            'stoch_k': momentum_analysis['current_stoch_k'],
            'stoch_d': momentum_analysis['current_stoch_d'],
            'stoch_zone': momentum_analysis['stoch_zone'],
            'recent_divergence': {
                'type': momentum_analysis['recent_divergence'].divergence_type,
                'strength': momentum_analysis['recent_divergence'].strength
            } if momentum_analysis['recent_divergence'] else None
        }

        return NavigationReport(
            timestamp=timestamp,
            current_price=current_price,
            wyckoff_phase=wyckoff_phase,
            sc_prediction=sc_prediction,
            bc_prediction=bc_prediction,
            nearest_support=confluence_analysis['nearest_support'],
            nearest_resistance=confluence_analysis['nearest_resistance'],
            all_confluence_zones=confluence_analysis['confluence_zones'],
            physics_metrics=physics_metrics,
            momentum_metrics=momentum_metrics,
            direction_probabilities=direction_probs,
            recommended_action=action,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            position_size_pct=size,
            overall_confidence=overall_confidence
        )

    def print_report(self, report: NavigationReport) -> str:
        """네비게이션 리포트 출력"""
        lines = []
        lines.append("=" * 70)
        lines.append("                    차트 네비게이션 리포트")
        lines.append("=" * 70)
        lines.append(f"시간: {report.timestamp}")
        lines.append(f"현재가: ${report.current_price:,.2f}")
        lines.append("")

        # 와이코프 분석
        lines.append("[현재 위치]")
        wp = report.wyckoff_phase
        lines.append(f"├── 구조: {wp.structure}")
        lines.append(f"├── 페이즈: {wp.phase}")
        lines.append(f"├── 진행도: {wp.progress*100:.0f}%")
        lines.append(f"├── 신뢰도: {wp.confidence*100:.0f}%")
        lines.append(f"└── {wp.description}")
        lines.append("")

        # SC 예측
        if report.sc_prediction:
            sc = report.sc_prediction
            lines.append("[SC/Spring 예측]")
            lines.append(f"├── 예상가: ${sc['price']:,.2f} ({sc['distance_pct']:+.2f}%)")
            lines.append(f"├── 컨플루언스 강도: {sc['strength']}개 레벨")
            lines.append(f"├── 물리학 신뢰도: {sc.get('physics_confidence', 0)*100:.0f}%")
            lines.append(f"├── 도달 확률: {sc.get('arrival_probability', 0)*100:.0f}%")
            lines.append(f"└── 예상 소요: {sc.get('estimated_bars', 0)}봉")
            lines.append("")

        # BC 예측
        if report.bc_prediction:
            bc = report.bc_prediction
            lines.append("[BC/UTAD 예측]")
            lines.append(f"├── 예상가: ${bc['price']:,.2f} ({bc['distance_pct']:+.2f}%)")
            lines.append(f"├── 컨플루언스 강도: {bc['strength']}개 레벨")
            lines.append(f"├── 물리학 신뢰도: {bc.get('physics_confidence', 0)*100:.0f}%")
            lines.append(f"├── 도달 확률: {bc.get('arrival_probability', 0)*100:.0f}%")
            lines.append(f"└── 예상 소요: {bc.get('estimated_bars', 0)}봉")
            lines.append("")

        # 물리학 지표
        pm = report.physics_metrics
        lines.append("[물리학 검증]")
        lines.append(f"├── 모멘텀: {pm['momentum']:.4f}")
        lines.append(f"├── 에너지: {pm['energy']:.2f}")
        lines.append(f"├── 엔트로피: {pm['entropy']:.2f}")
        lines.append(f"├── 평균회귀력: {pm['mean_reversion_force']:.2f}")
        lines.append(f"├── 추세관성: {pm['trend_inertia']:.2f}")
        lines.append(f"└── 지배사이클: {pm['dominant_cycle']}봉")
        lines.append("")

        # 모멘텀 지표
        mm = report.momentum_metrics
        lines.append("[모멘텀 지표]")
        lines.append(f"├── RSI: {mm['rsi']:.1f}")
        lines.append(f"├── Stoch RSI K: {mm['stoch_k']:.1f}")
        lines.append(f"├── Stoch RSI D: {mm['stoch_d']:.1f}")
        lines.append(f"├── 상태: {mm['stoch_zone']}")
        if mm['recent_divergence']:
            lines.append(f"└── 다이버전스: {mm['recent_divergence']['type']} (강도: {mm['recent_divergence']['strength']:.2f})")
        else:
            lines.append(f"└── 다이버전스: 없음")
        lines.append("")

        # 방향 확률
        dp = report.direction_probabilities
        lines.append("[방향 확률]")
        lines.append(f"├── 상승: {dp['up']*100:.1f}%")
        lines.append(f"├── 하락: {dp['down']*100:.1f}%")
        lines.append(f"└── 횡보: {dp['neutral']*100:.1f}%")
        lines.append("")

        # 추천 액션
        lines.append("[추천 액션]")
        lines.append(f"★ {report.recommended_action}")
        if report.entry_price:
            lines.append(f"├── 진입가: ${report.entry_price:,.2f}")
        if report.stop_loss:
            lines.append(f"├── 손절: ${report.stop_loss:,.2f}")
        if report.take_profit:
            lines.append(f"├── 목표: ${report.take_profit:,.2f}")
        lines.append(f"└── 포지션: {report.position_size_pct*100:.1f}%")
        lines.append("")

        lines.append(f"전체 신뢰도: {report.overall_confidence*100:.0f}%")
        lines.append("=" * 70)

        return "\n".join(lines)
