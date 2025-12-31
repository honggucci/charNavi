"""
WPCN Probability Models V2
확률 모델 모듈 (GPT 피드백 반영)

P2: 점수 기반 → 확률 기반 시스템 변환

FIXED:
- (A) Finite Horizon: Monte Carlo 기반 시간 제한 확률
- (B) 단위 통일: 로그 수익률 공간
- (C) Drift 최소화: 클램프 적용
- (D) 수치 안정: expm1 사용
- (E) EV 정의: R-unit + 비용 포함
- (F) 캘리브레이션: 3-class (TP/SL/Timeout)
"""
from wpcn._05_probability.barrier import (
    BarrierProbability,
    ExitType,
    calculate_barrier_probability,
    monte_carlo_barrier_probability,
    first_passage_probability,
    first_passage_probability_analytic,
    estimate_drift_from_momentum,
    estimate_drift_from_indicators,
    estimate_drift_minimal,
    estimate_volatility_from_atr,
    estimate_momentum_from_price,  # FIXED (P0-4): 가격 기반 모멘텀
    calculate_ev_with_costs,
    quick_probability
)
from wpcn._05_probability.calibration import (
    ProbabilityCalibrator,
    CalibrationResult,
    PredictionRecord,
    ExitOutcome,
    reliability_diagram_data
)

__all__ = [
    # Barrier Probability
    'BarrierProbability',
    'ExitType',
    'calculate_barrier_probability',
    'monte_carlo_barrier_probability',
    'first_passage_probability',
    'first_passage_probability_analytic',
    'estimate_drift_from_momentum',
    'estimate_drift_from_indicators',
    'estimate_drift_minimal',
    'estimate_volatility_from_atr',
    'estimate_momentum_from_price',  # FIXED (P0-4)
    'calculate_ev_with_costs',
    'quick_probability',
    # Calibration
    'ProbabilityCalibrator',
    'CalibrationResult',
    'PredictionRecord',
    'ExitOutcome',
    'reliability_diagram_data'
]
