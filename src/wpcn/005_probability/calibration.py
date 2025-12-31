"""
Probability Calibration Module V2
확률 캘리브레이션 모듈 (GPT 피드백 반영)

FIXED (F): 3-class 캘리브레이션 (TP/SL/Timeout)
- 기존: 2-class (TP=1, SL=0)
- 변경: 3-class (TP=1, SL=0, Timeout=별도 추적)

목적:
- P(TP) = 0.7 예측 시, 실제로 70%가 TP 도달하는지 검증
- P(Timeout)도 별도로 추적하여 모델 정확도 향상
- 과신/과소신 보정 계수 계산
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExitOutcome(Enum):
    """청산 결과 (3-class)"""
    TP = "tp"           # Take Profit 도달
    SL = "sl"           # Stop Loss 도달
    TIMEOUT = "timeout" # 시간초과 청산
    LIQUIDATION = "liquidation"  # 청산


@dataclass
class CalibrationResult:
    """
    캘리브레이션 결과 (3-class 지원)

    Attributes:
        brier_score: Brier Score (낮을수록 좋음, 0~1)
        brier_score_3class: 3-class Brier Score
        calibration_error: 평균 캘리브레이션 오차
        overconfidence: 과신 정도 (양수 = 과신, 음수 = 과소신)
        bin_accuracy: 확률 구간별 실제 적중률
        n_samples: 샘플 수
        outcome_distribution: 결과 분포 (TP/SL/Timeout/Liq 비율)
    """
    brier_score: float
    brier_score_3class: float
    calibration_error: float
    overconfidence: float
    bin_accuracy: Dict[str, Dict]
    n_samples: int
    outcome_distribution: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            'brier_score': self.brier_score,
            'brier_score_3class': self.brier_score_3class,
            'calibration_error': self.calibration_error,
            'overconfidence': self.overconfidence,
            'bin_accuracy': self.bin_accuracy,
            'n_samples': self.n_samples,
            'outcome_distribution': self.outcome_distribution
        }


@dataclass
class PredictionRecord:
    """
    단일 예측 기록 (3-class 지원)

    FIXED (F): exit_reason 필드 추가
    """
    predicted_p_tp: float       # 예측된 TP 확률
    predicted_p_sl: float       # 예측된 SL 확률
    predicted_p_timeout: float  # 예측된 Timeout 확률
    actual_outcome: str         # 실제 결과: "tp", "sl", "timeout", "liquidation"
    timestamp: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def actual_outcome_binary(self) -> int:
        """2-class 변환 (TP=1, else=0)"""
        return 1 if self.actual_outcome == "tp" else 0


class ProbabilityCalibrator:
    """
    확률 캘리브레이션 관리자 V2 (3-class 지원)

    FIXED (F): Timeout을 별도로 추적

    Usage:
        calibrator = ProbabilityCalibrator()

        # 완료된 트레이드 기록
        calibrator.add_completed_record(
            predicted_p_tp=0.6,
            predicted_p_sl=0.3,
            predicted_p_timeout=0.1,
            exit_reason="tp"
        )

        # 분석
        result = calibrator.analyze()
    """

    def __init__(self, n_bins: int = 10):
        """
        Args:
            n_bins: 캘리브레이션 구간 수
        """
        self.n_bins = n_bins
        self.predictions: Dict[str, Dict[str, float]] = {}  # trade_id -> {p_tp, p_sl, p_timeout}
        self.outcomes: Dict[str, str] = {}                   # trade_id -> outcome
        self.records: List[PredictionRecord] = []

    def record_prediction(
        self,
        p_tp: float,
        trade_id: str,
        p_sl: float = 0.0,
        p_timeout: float = 0.0,
        timestamp: Optional[str] = None,
        **metadata
    ) -> None:
        """
        예측 기록 (3-class)

        Args:
            p_tp: TP 확률
            p_sl: SL 확률
            p_timeout: Timeout 확률
            trade_id: 트레이드 식별자
            timestamp: 타임스탬프
            **metadata: 추가 메타데이터
        """
        # 정규화
        total = p_tp + p_sl + p_timeout
        if total > 0:
            p_tp /= total
            p_sl /= total
            p_timeout /= total
        else:
            p_tp = p_sl = 0.5
            p_timeout = 0.0

        self.predictions[trade_id] = {
            'p_tp': np.clip(p_tp, 0.0, 1.0),
            'p_sl': np.clip(p_sl, 0.0, 1.0),
            'p_timeout': np.clip(p_timeout, 0.0, 1.0)
        }

        logger.debug(f"Recorded prediction: {trade_id} -> p_tp={p_tp:.3f}")

    def record_outcome(
        self,
        trade_id: str,
        exit_reason: str
    ) -> None:
        """
        결과 기록 (3-class)

        Args:
            trade_id: 트레이드 식별자
            exit_reason: 청산 사유 ("tp", "sl", "timeout", "liquidation")
        """
        if trade_id not in self.predictions:
            logger.warning(f"No prediction found for {trade_id}")
            return

        # 유효한 exit_reason 확인
        valid_outcomes = ["tp", "sl", "timeout", "liquidation"]
        exit_reason = exit_reason.lower()
        if exit_reason not in valid_outcomes:
            logger.warning(f"Invalid exit_reason: {exit_reason}")
            exit_reason = "sl"  # 기본값

        self.outcomes[trade_id] = exit_reason

        # 완료된 레코드 저장
        pred = self.predictions[trade_id]
        self.records.append(PredictionRecord(
            predicted_p_tp=pred['p_tp'],
            predicted_p_sl=pred['p_sl'],
            predicted_p_timeout=pred['p_timeout'],
            actual_outcome=exit_reason
        ))

        logger.debug(f"Recorded outcome: {trade_id} -> {exit_reason}")

    def add_completed_record(
        self,
        predicted_prob: float,
        hit_tp: bool = False,
        exit_reason: str = None,
        predicted_p_sl: float = None,
        predicted_p_timeout: float = None,
        timestamp: Optional[str] = None,
        **metadata
    ) -> None:
        """
        완료된 트레이드 직접 추가

        FIXED (F): exit_reason으로 3-class 지원

        Args:
            predicted_prob: 예측된 TP 확률
            hit_tp: TP 도달 여부 (레거시 호환)
            exit_reason: 청산 사유 ("tp", "sl", "timeout", "liquidation")
            predicted_p_sl: 예측된 SL 확률
            predicted_p_timeout: 예측된 Timeout 확률
            timestamp: 타임스탬프
            **metadata: 추가 정보
        """
        p_tp = np.clip(predicted_prob, 0.0, 1.0)

        # exit_reason 결정
        if exit_reason is not None:
            outcome = exit_reason.lower()
        else:
            outcome = "tp" if hit_tp else "sl"

        # p_sl, p_timeout 추정 (없으면)
        if predicted_p_sl is None or predicted_p_timeout is None:
            # 나머지를 균등 배분
            remaining = 1.0 - p_tp
            p_sl = remaining * 0.8  # 대부분 SL
            p_timeout = remaining * 0.2
        else:
            p_sl = predicted_p_sl
            p_timeout = predicted_p_timeout

        self.records.append(PredictionRecord(
            predicted_p_tp=p_tp,
            predicted_p_sl=p_sl,
            predicted_p_timeout=p_timeout,
            actual_outcome=outcome,
            timestamp=timestamp,
            metadata=metadata
        ))

    def analyze(self) -> Optional[CalibrationResult]:
        """
        캘리브레이션 분석 (3-class)

        Returns:
            CalibrationResult or None if insufficient data
        """
        if len(self.records) < 10:
            logger.warning(f"Insufficient data for calibration: {len(self.records)} records")
            return None

        # 데이터 추출
        p_tp_array = np.array([r.predicted_p_tp for r in self.records])
        p_sl_array = np.array([r.predicted_p_sl for r in self.records])
        p_timeout_array = np.array([r.predicted_p_timeout for r in self.records])
        outcomes = [r.actual_outcome for r in self.records]
        outcomes_binary = np.array([r.actual_outcome_binary for r in self.records])

        # 결과 분포
        outcome_counts = {
            'tp': sum(1 for o in outcomes if o == 'tp'),
            'sl': sum(1 for o in outcomes if o == 'sl'),
            'timeout': sum(1 for o in outcomes if o == 'timeout'),
            'liquidation': sum(1 for o in outcomes if o == 'liquidation')
        }
        total = len(outcomes)
        outcome_distribution = {k: v / total for k, v in outcome_counts.items()}

        # 2-class Brier Score (TP vs not-TP)
        brier_2class = self._brier_score(p_tp_array, outcomes_binary)

        # 3-class Brier Score
        brier_3class = self._brier_score_multiclass(
            p_tp_array, p_sl_array, p_timeout_array, outcomes
        )

        # Calibration Error (TP 확률 기준)
        cal_error, bin_accuracy, overconfidence = self._calibration_analysis(
            p_tp_array, outcomes_binary
        )

        return CalibrationResult(
            brier_score=brier_2class,
            brier_score_3class=brier_3class,
            calibration_error=cal_error,
            overconfidence=overconfidence,
            bin_accuracy=bin_accuracy,
            n_samples=len(self.records),
            outcome_distribution=outcome_distribution
        )

    def _brier_score(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray
    ) -> float:
        """
        2-class Brier Score

        BS = (1/N) * Σ(p_i - o_i)²

        낮을수록 좋음. 랜덤 = 0.25, 완벽 = 0.0
        """
        return float(np.mean((predictions - outcomes) ** 2))

    def _brier_score_multiclass(
        self,
        p_tp: np.ndarray,
        p_sl: np.ndarray,
        p_timeout: np.ndarray,
        outcomes: List[str]
    ) -> float:
        """
        3-class Brier Score

        FIXED (F): 3-class 확률 검증

        BS = (1/N) * Σ Σ_k (p_ik - y_ik)²
        """
        n = len(outcomes)
        total_error = 0.0

        for i, outcome in enumerate(outcomes):
            # One-hot encoding
            y_tp = 1.0 if outcome == 'tp' else 0.0
            y_sl = 1.0 if outcome in ['sl', 'liquidation'] else 0.0
            y_timeout = 1.0 if outcome == 'timeout' else 0.0

            # Squared errors
            total_error += (p_tp[i] - y_tp) ** 2
            total_error += (p_sl[i] - y_sl) ** 2
            total_error += (p_timeout[i] - y_timeout) ** 2

        return total_error / (n * 3)  # 3 classes

    def _calibration_analysis(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray
    ) -> Tuple[float, Dict[str, Dict], float]:
        """
        캘리브레이션 분석

        Returns:
            (calibration_error, bin_accuracy, overconfidence)
        """
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_accuracy = {}
        errors = []
        confidence_gaps = []

        for i in range(self.n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            mask = (predictions >= low) & (predictions < high)

            if not mask.any():
                continue

            bin_preds = predictions[mask]
            bin_outcomes = outcomes[mask]

            mean_pred = float(np.mean(bin_preds))
            actual_rate = float(np.mean(bin_outcomes))

            bin_name = f"{low:.1f}-{high:.1f}"
            bin_accuracy[bin_name] = {
                'predicted': mean_pred,
                'actual': actual_rate,
                'count': int(mask.sum()),
                'gap': mean_pred - actual_rate
            }

            error = abs(mean_pred - actual_rate)
            errors.append(error)

            # 과신: 예측 > 실제
            confidence_gaps.append(mean_pred - actual_rate)

        cal_error = float(np.mean(errors)) if errors else 0.0
        overconfidence = float(np.mean(confidence_gaps)) if confidence_gaps else 0.0

        return cal_error, bin_accuracy, overconfidence

    def get_correction_factor(self) -> float:
        """
        보정 계수 계산

        과신 시 < 1.0, 과소신 시 > 1.0

        사용법:
            corrected_prob = raw_prob * correction_factor
        """
        result = self.analyze()
        if result is None:
            return 1.0

        # 과신이면 확률을 낮춰야 함
        # overconfidence = 0.1 이면 평균적으로 10%p 과신
        # correction = 1 - overconfidence
        return max(0.5, min(1.5, 1.0 - result.overconfidence))

    def get_timeout_rate(self) -> float:
        """Timeout 비율 반환"""
        if not self.records:
            return 0.0
        timeout_count = sum(1 for r in self.records if r.actual_outcome == 'timeout')
        return timeout_count / len(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        """레코드를 DataFrame으로 변환"""
        if not self.records:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'predicted_p_tp': r.predicted_p_tp,
                'predicted_p_sl': r.predicted_p_sl,
                'predicted_p_timeout': r.predicted_p_timeout,
                'actual_outcome': r.actual_outcome,
                'actual_outcome_binary': r.actual_outcome_binary,
                'timestamp': r.timestamp,
                **r.metadata
            }
            for r in self.records
        ])

    def clear(self) -> None:
        """모든 기록 초기화"""
        self.predictions.clear()
        self.outcomes.clear()
        self.records.clear()

    def __len__(self) -> int:
        return len(self.records)


def reliability_diagram_data(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """
    Reliability Diagram 데이터 생성

    Returns:
        {
            'bin_centers': 구간 중심값,
            'bin_accuracy': 구간별 실제 적중률,
            'bin_counts': 구간별 샘플 수
        }
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracy = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (predictions >= low) & (predictions < high)

        if mask.sum() > 0:
            bin_centers.append((low + high) / 2)
            bin_accuracy.append(float(np.mean(outcomes[mask])))
            bin_counts.append(int(mask.sum()))

    return {
        'bin_centers': np.array(bin_centers),
        'bin_accuracy': np.array(bin_accuracy),
        'bin_counts': np.array(bin_counts)
    }
