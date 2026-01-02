"""
Sensitivity Analyzer - 파라미터 민감도 분석
============================================

최적화된 파라미터의 안정성을 검증합니다.

핵심 기능:
1. 파라미터 ±10% 변화 시 성능 변동폭 측정
2. 안정적 파라미터 vs 과적합 파라미터 구분
3. sensitivity_score 계산 (0~1, 낮을수록 안정적)

사용법:
    from wpcn._08_tuning.sensitivity_analyzer import SensitivityAnalyzer

    analyzer = SensitivityAnalyzer(backtest_func)
    result = analyzer.analyze(df, best_params, timeframe="15m")

    print(f"Sensitivity Score: {result.overall_score:.3f}")
    for param, info in result.param_sensitivities.items():
        print(f"  {param}: {info['sensitivity']:.3f}")
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, Optional, Tuple
import numpy as np
import pandas as pd

from .param_schema import (
    TUNABLE_PARAMS_MINUTES,
    TUNABLE_PARAMS_RATIO,
    MINUTES_TO_BARS_KEY_MAP,
    minutes_to_bars,
)


@dataclass
class ParamSensitivity:
    """단일 파라미터의 민감도 분석 결과"""
    param_name: str
    base_value: float
    perturbations: List[float]  # [-10%, -5%, +5%, +10%]
    scores: List[float]         # 각 perturbation에서의 성능
    base_score: float
    sensitivity: float          # 성능 변동률 (0~1)
    is_stable: bool             # sensitivity < 0.3

    def to_dict(self) -> Dict:
        return {
            "param_name": self.param_name,
            "base_value": self.base_value,
            "base_score": self.base_score,
            "perturbations": self.perturbations,
            "scores": self.scores,
            "sensitivity": self.sensitivity,
            "is_stable": self.is_stable,
        }


@dataclass
class SensitivityResult:
    """전체 민감도 분석 결과"""
    param_sensitivities: Dict[str, ParamSensitivity]
    overall_score: float        # 평균 민감도 (0~1, 낮을수록 좋음)
    stable_params: List[str]    # 안정적인 파라미터
    unstable_params: List[str]  # 불안정한 파라미터
    base_score: float           # 기본 파라미터 성능

    def to_dict(self) -> Dict:
        return {
            "overall_score": self.overall_score,
            "base_score": self.base_score,
            "stable_params": self.stable_params,
            "unstable_params": self.unstable_params,
            "param_details": {
                k: v.to_dict() for k, v in self.param_sensitivities.items()
            }
        }

    @property
    def is_robust(self) -> bool:
        """전반적으로 안정적인지 (overall_score < 0.3)"""
        return self.overall_score < 0.3

    @property
    def stability_grade(self) -> str:
        """안정성 등급"""
        if self.overall_score < 0.15:
            return "A"  # 매우 안정적
        elif self.overall_score < 0.25:
            return "B"  # 안정적
        elif self.overall_score < 0.35:
            return "C"  # 보통
        elif self.overall_score < 0.50:
            return "D"  # 불안정
        else:
            return "F"  # 매우 불안정 (과적합 의심)


class SensitivityAnalyzer:
    """
    파라미터 민감도 분석기

    각 파라미터를 ±5%, ±10% 변화시키며 성능 변동을 측정합니다.
    변동이 클수록 해당 파라미터에 과적합되었을 가능성이 높습니다.
    """

    # 기본 perturbation 비율
    DEFAULT_PERTURBATIONS = [-0.10, -0.05, 0.05, 0.10]

    # 민감도 임계값
    STABLE_THRESHOLD = 0.30     # 이하면 안정적
    UNSTABLE_THRESHOLD = 0.50   # 이상이면 불안정

    def __init__(
        self,
        backtest_func: Callable[[pd.DataFrame, Dict], Dict],
        score_key: str = "sharpe_ratio",
        perturbations: List[float] = None,
        min_score: float = -10.0,
    ):
        """
        Args:
            backtest_func: 백테스트 함수 (df, params) -> {"sharpe_ratio": ..., ...}
            score_key: 성능 지표 키 (기본: sharpe_ratio)
            perturbations: 변동 비율 리스트 (기본: [-10%, -5%, +5%, +10%])
            min_score: 최소 유효 점수 (이하면 무효 처리)
        """
        self.backtest_func = backtest_func
        self.score_key = score_key
        self.perturbations = perturbations or self.DEFAULT_PERTURBATIONS
        self.min_score = min_score

    def _get_score(self, result: Dict) -> float:
        """백테스트 결과에서 점수 추출"""
        score = result.get(self.score_key, self.min_score)
        if score is None or np.isnan(score) or np.isinf(score):
            return self.min_score
        return float(score)

    def _get_tunable_params(self, params: Dict) -> List[str]:
        """튜닝 가능한 파라미터 키 추출"""
        tunable = []

        # 분 단위 파라미터 (저장 형식)
        for key in TUNABLE_PARAMS_MINUTES.keys():
            if key in params:
                tunable.append(key)

        # 봉 단위 파라미터 (런타임 형식)
        for bars_key in MINUTES_TO_BARS_KEY_MAP.values():
            if bars_key in params and bars_key != "_pivot_bars":
                tunable.append(bars_key)

        # pivot_lr 특수 처리
        if "pivot_lr" in params:
            tunable.append("pivot_lr")

        # 비율 파라미터
        for key in TUNABLE_PARAMS_RATIO.keys():
            if key in params:
                tunable.append(key)

        return list(set(tunable))

    def _perturb_param(
        self,
        params: Dict,
        param_key: str,
        perturbation: float
    ) -> Dict:
        """단일 파라미터에 perturbation 적용"""
        perturbed = params.copy()
        base_value = params[param_key]

        # 정수형 파라미터 (봉 수, 분 수)
        if isinstance(base_value, int) or param_key.endswith(("_minutes", "_bars", "_lr")):
            new_value = int(round(base_value * (1 + perturbation)))
            new_value = max(1, new_value)  # 최소 1
        else:
            new_value = base_value * (1 + perturbation)
            # 비율 파라미터 범위 제한
            if param_key in ("tp_pct", "sl_pct", "x_atr", "m_bw", "F_min"):
                new_value = max(0.001, min(1.0, new_value))

        perturbed[param_key] = new_value
        return perturbed

    def analyze_param(
        self,
        df: pd.DataFrame,
        params: Dict,
        param_key: str,
        base_score: float = None
    ) -> ParamSensitivity:
        """
        단일 파라미터 민감도 분석

        Args:
            df: OHLCV 데이터
            params: 기본 파라미터
            param_key: 분석할 파라미터 키
            base_score: 기본 점수 (None이면 계산)

        Returns:
            ParamSensitivity
        """
        base_value = params[param_key]

        # 기본 점수 계산
        if base_score is None:
            base_result = self.backtest_func(df, params)
            base_score = self._get_score(base_result)

        # 각 perturbation에서 점수 계산
        scores = []
        for pert in self.perturbations:
            perturbed_params = self._perturb_param(params, param_key, pert)
            try:
                result = self.backtest_func(df, perturbed_params)
                score = self._get_score(result)
            except Exception:
                score = self.min_score
            scores.append(score)

        # 민감도 계산: 성능 변동률
        # base_score 대비 변동 정도 (정규화)
        if abs(base_score) > 0.001:
            deviations = [abs(s - base_score) / abs(base_score) for s in scores]
        else:
            deviations = [abs(s - base_score) for s in scores]

        sensitivity = float(np.mean(deviations))

        # 민감도가 너무 크면 1로 제한
        sensitivity = min(1.0, sensitivity)

        return ParamSensitivity(
            param_name=param_key,
            base_value=base_value,
            perturbations=self.perturbations,
            scores=scores,
            base_score=base_score,
            sensitivity=sensitivity,
            is_stable=sensitivity < self.STABLE_THRESHOLD
        )

    def analyze(
        self,
        df: pd.DataFrame,
        params: Dict,
        param_keys: List[str] = None,
        verbose: bool = True
    ) -> SensitivityResult:
        """
        전체 파라미터 민감도 분석

        Args:
            df: OHLCV 데이터
            params: 최적화된 파라미터
            param_keys: 분석할 파라미터 키 (None이면 자동 감지)
            verbose: 진행 상황 출력

        Returns:
            SensitivityResult
        """
        if param_keys is None:
            param_keys = self._get_tunable_params(params)

        if verbose:
            print(f"[Sensitivity] Analyzing {len(param_keys)} parameters...")

        # 기본 점수 계산
        base_result = self.backtest_func(df, params)
        base_score = self._get_score(base_result)

        if verbose:
            print(f"  Base score ({self.score_key}): {base_score:.4f}")

        # 각 파라미터 분석
        sensitivities = {}
        for i, key in enumerate(param_keys):
            if key not in params:
                continue

            if verbose:
                print(f"  [{i+1}/{len(param_keys)}] {key}: ", end="")

            try:
                sens = self.analyze_param(df, params, key, base_score)
                sensitivities[key] = sens

                if verbose:
                    status = "stable" if sens.is_stable else "UNSTABLE"
                    print(f"sensitivity={sens.sensitivity:.3f} ({status})")

            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")

        # 안정/불안정 파라미터 분류
        stable_params = [k for k, v in sensitivities.items() if v.is_stable]
        unstable_params = [k for k, v in sensitivities.items() if not v.is_stable]

        # 전체 민감도 점수
        if sensitivities:
            overall_score = float(np.mean([v.sensitivity for v in sensitivities.values()]))
        else:
            overall_score = 0.0

        result = SensitivityResult(
            param_sensitivities=sensitivities,
            overall_score=overall_score,
            stable_params=stable_params,
            unstable_params=unstable_params,
            base_score=base_score
        )

        if verbose:
            print(f"\n[Sensitivity] Overall: {overall_score:.3f} (Grade: {result.stability_grade})")
            print(f"  Stable: {len(stable_params)}, Unstable: {len(unstable_params)}")

        return result

    def quick_check(
        self,
        df: pd.DataFrame,
        params: Dict,
        top_n: int = 5
    ) -> Dict[str, float]:
        """
        빠른 민감도 체크 (주요 파라미터만)

        Args:
            df: OHLCV 데이터
            params: 파라미터
            top_n: 체크할 파라미터 수

        Returns:
            {param_name: sensitivity}
        """
        # 주요 파라미터 우선순위
        priority_keys = [
            "box_L", "box_window_minutes",
            "m_freeze", "freeze_minutes",
            "x_atr", "tp_pct", "sl_pct"
        ]

        check_keys = []
        for key in priority_keys:
            if key in params and len(check_keys) < top_n:
                check_keys.append(key)

        result = self.analyze(df, params, param_keys=check_keys, verbose=False)

        return {
            k: v.sensitivity
            for k, v in result.param_sensitivities.items()
        }


def compute_sensitivity_score(
    sensitivities: Dict[str, float],
    weights: Dict[str, float] = None
) -> float:
    """
    민감도 점수 계산 (가중 평균)

    Args:
        sensitivities: {param_name: sensitivity}
        weights: {param_name: weight} (없으면 균등 가중)

    Returns:
        종합 민감도 점수 (0~1)
    """
    if not sensitivities:
        return 0.0

    if weights is None:
        return float(np.mean(list(sensitivities.values())))

    weighted_sum = 0.0
    total_weight = 0.0

    for param, sens in sensitivities.items():
        w = weights.get(param, 1.0)
        weighted_sum += sens * w
        total_weight += w

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight


def classify_params_by_stability(
    result: SensitivityResult
) -> Tuple[List[str], List[str], List[str]]:
    """
    파라미터를 안정성에 따라 분류

    Returns:
        (very_stable, moderate, unstable)
    """
    very_stable = []  # sensitivity < 0.15
    moderate = []     # 0.15 <= sensitivity < 0.35
    unstable = []     # sensitivity >= 0.35

    for param, sens in result.param_sensitivities.items():
        if sens.sensitivity < 0.15:
            very_stable.append(param)
        elif sens.sensitivity < 0.35:
            moderate.append(param)
        else:
            unstable.append(param)

    return very_stable, moderate, unstable
