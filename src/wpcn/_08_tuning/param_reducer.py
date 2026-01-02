"""
Parameter Reducer - 차원 축소 모듈 (P1: 과적합 억제)
===================================================

Sensitivity 분석 기반으로 파라미터 공간을 자동 축소합니다.

핵심 기능:
1. Sensitivity 기반 자동 freeze: 불안정한 파라미터 고정
2. 재파라미터화: sl_pct/tp_pct → RRR + sl_pct
3. 다음 주 param_space 자동 생성

Usage:
    from wpcn._08_tuning.param_reducer import ParamReducer

    reducer = ParamReducer()

    # Sensitivity 결과로 축소된 param_space 생성
    reduced_space = reducer.reduce_from_sensitivity(
        full_space=V8_PARAM_SPACE,
        sensitivity_result=sens_result,
        freeze_unstable=True,
        fixed_values={"box_L": 96}  # 불안정 파라미터 고정값
    )

    # 재파라미터화 (sl/tp → RRR)
    reparam_space = reducer.reparameterize_risk_reward(full_space)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import json
from pathlib import Path
from datetime import datetime


@dataclass
class ReductionConfig:
    """
    차원 축소 설정

    NOTE: Sensitivity = instability (불안정도), NOT importance (중요도).
    높은 sensitivity = 파라미터 변화에 성능이 민감하게 반응 = 불안정 = freeze 대상.
    상세 정의는 sensitivity_analyzer.py 참조.
    """
    # Freeze 임계값 (sensitivity = instability 기준)
    freeze_sensitivity_threshold: float = 0.4  # 이상이면 freeze (너무 불안정)
    keep_sensitivity_threshold: float = 0.25   # 이하면 반드시 유지 (충분히 안정)

    # 최대 동적 파라미터 수
    max_dynamic_params: int = 6

    # 재파라미터화 옵션
    use_rrr_reparameterization: bool = True  # sl/tp → RRR 변환
    use_box_ratio_reparameterization: bool = True  # m_freeze를 box_L 비율로

    # 고정 파라미터 (항상 freeze)
    always_freeze: List[str] = field(default_factory=lambda: [
        "N_reclaim", "N_fill", "F_min", "m_bw"
    ])


@dataclass
class ReducedParamSpace:
    """축소된 파라미터 공간"""
    dynamic_params: Dict[str, Tuple]    # 최적화할 파라미터
    frozen_params: Dict[str, Any]       # 고정된 파라미터
    reparameterized: Dict[str, str]     # 재파라미터화 매핑
    reduction_ratio: float               # 축소율 (0~1)
    source_sensitivity: Optional[Dict]   # 원본 sensitivity 결과

    def to_optimizer_space(self) -> Dict[str, Tuple]:
        """옵티마이저용 공간 반환"""
        return self.dynamic_params.copy()

    def to_dict(self) -> Dict:
        return {
            "dynamic_params": self.dynamic_params,
            "frozen_params": self.frozen_params,
            "reparameterized": self.reparameterized,
            "reduction_ratio": self.reduction_ratio,
            "n_dynamic": len(self.dynamic_params),
            "n_frozen": len(self.frozen_params),
        }

    def get_full_params(self, optimized: Dict) -> Dict:
        """최적화된 파라미터 + frozen 파라미터 합치기"""
        full = self.frozen_params.copy()
        full.update(optimized)

        # 재파라미터화 역변환
        if "rrr" in full and "sl_pct" in full:
            # RRR = tp / sl → tp = RRR * sl
            full["tp_pct"] = full["rrr"] * full["sl_pct"]
            del full["rrr"]

        if "m_freeze_ratio" in full and "box_L" in full:
            # m_freeze_ratio = m_freeze / box_L → m_freeze = ratio * box_L
            full["m_freeze"] = int(full["m_freeze_ratio"] * full["box_L"])
            del full["m_freeze_ratio"]

        return full


class ParamReducer:
    """
    파라미터 차원 축소기

    Sensitivity 분석 결과를 기반으로:
    1. 불안정한 파라미터를 freeze
    2. 관련 파라미터를 재파라미터화
    3. 최적화 공간 차원 감소
    """

    def __init__(self, config: ReductionConfig = None):
        self.config = config or ReductionConfig()

    def reduce_from_sensitivity(
        self,
        full_space: Dict[str, Tuple],
        sensitivity_result: Any,  # SensitivityResult
        fixed_values: Dict[str, Any] = None,
    ) -> ReducedParamSpace:
        """
        Sensitivity 결과 기반 차원 축소

        Args:
            full_space: 전체 파라미터 공간
            sensitivity_result: SensitivityResult 객체
            fixed_values: 불안정 파라미터 고정값

        Returns:
            ReducedParamSpace
        """
        cfg = self.config
        fixed_values = fixed_values or {}

        dynamic_params = {}
        frozen_params = {}
        reparameterized = {}

        # 1. Sensitivity 기반 분류
        param_sensitivities = {}
        if hasattr(sensitivity_result, 'param_sensitivities'):
            param_sensitivities = {
                k: v.sensitivity
                for k, v in sensitivity_result.param_sensitivities.items()
            }

        for param, spec in full_space.items():
            # 항상 freeze 리스트 체크
            if param in cfg.always_freeze:
                frozen_params[param] = fixed_values.get(param, self._get_default_value(spec))
                continue

            sensitivity = param_sensitivities.get(param, 0.0)

            # 매우 안정적인 파라미터는 반드시 유지 (keep_sensitivity_threshold 사용)
            if sensitivity <= cfg.keep_sensitivity_threshold:
                dynamic_params[param] = spec
                continue

            # 불안정 파라미터 freeze
            if sensitivity >= cfg.freeze_sensitivity_threshold:
                if param in fixed_values:
                    frozen_params[param] = fixed_values[param]
                else:
                    # 스펙 중앙값으로 고정
                    frozen_params[param] = self._get_median_value(spec)
                continue

            # 중간 영역 (keep < sens < freeze): 동적으로 유지
            dynamic_params[param] = spec

        # 2. 재파라미터화 적용
        if cfg.use_rrr_reparameterization:
            dynamic_params, reparameterized = self._apply_rrr_reparam(
                dynamic_params, frozen_params, reparameterized
            )

        if cfg.use_box_ratio_reparameterization:
            dynamic_params, reparameterized = self._apply_box_ratio_reparam(
                dynamic_params, frozen_params, reparameterized
            )

        # 3. 최대 차원 제한
        if len(dynamic_params) > cfg.max_dynamic_params:
            # Sensitivity 낮은 순으로 정렬, 상위만 유지
            sorted_params = sorted(
                dynamic_params.keys(),
                key=lambda p: param_sensitivities.get(p, 0.0)
            )

            # 상위 max_dynamic_params개만 유지
            keep_params = set(sorted_params[:cfg.max_dynamic_params])

            for param in list(dynamic_params.keys()):
                if param not in keep_params:
                    frozen_params[param] = self._get_median_value(dynamic_params[param])
                    del dynamic_params[param]

        # 축소율 계산
        original_dim = len(full_space)
        reduced_dim = len(dynamic_params)
        reduction_ratio = 1 - (reduced_dim / original_dim) if original_dim > 0 else 0

        return ReducedParamSpace(
            dynamic_params=dynamic_params,
            frozen_params=frozen_params,
            reparameterized=reparameterized,
            reduction_ratio=reduction_ratio,
            source_sensitivity=param_sensitivities
        )

    def _apply_rrr_reparam(
        self,
        dynamic: Dict,
        frozen: Dict,
        reparam: Dict
    ) -> Tuple[Dict, Dict]:
        """
        sl/tp 재파라미터화: tp_pct → RRR (Risk Reward Ratio)

        변환: tp_pct = RRR * sl_pct
        장점: RRR만 최적화하면 tp/sl 비율 자동 유지
        """
        if "tp_pct" in dynamic and "sl_pct" in dynamic:
            # sl_pct는 유지, tp_pct를 RRR로 대체
            tp_spec = dynamic["tp_pct"]
            sl_spec = dynamic["sl_pct"]

            # RRR 범위 계산: tp_min/sl_max ~ tp_max/sl_min
            rrr_min = tp_spec[0] / sl_spec[1] if sl_spec[1] > 0 else 1.0
            rrr_max = tp_spec[1] / sl_spec[0] if sl_spec[0] > 0 else 3.0

            # 합리적인 범위로 제한
            rrr_min = max(1.0, rrr_min)
            rrr_max = min(5.0, rrr_max)

            # 역전 방지: min > max 케이스 처리
            if rrr_min > rrr_max:
                rrr_min, rrr_max = 1.5, 2.5  # fallback to safe default

            dynamic["rrr"] = (rrr_min, rrr_max)
            del dynamic["tp_pct"]
            reparam["tp_pct"] = "rrr * sl_pct"

        return dynamic, reparam

    def _apply_box_ratio_reparam(
        self,
        dynamic: Dict,
        frozen: Dict,
        reparam: Dict
    ) -> Tuple[Dict, Dict]:
        """
        m_freeze 재파라미터화: m_freeze → m_freeze_ratio (box_L 대비 비율)

        변환: m_freeze = m_freeze_ratio * box_L
        장점: box_L이 변해도 비율 일정 유지
        """
        if "m_freeze" in dynamic and "box_L" in dynamic:
            # m_freeze를 비율로 대체
            m_spec = dynamic["m_freeze"]
            box_spec = dynamic["box_L"]

            # 비율 범위 계산: m_min/box_max ~ m_max/box_min
            ratio_min = m_spec[0] / box_spec[1] if box_spec[1] > 0 else 0.1
            ratio_max = m_spec[1] / box_spec[0] if box_spec[0] > 0 else 0.5

            # 제약조건 반영: 0.1 ~ 0.5
            ratio_min = max(0.1, ratio_min)
            ratio_max = min(0.5, ratio_max)

            # 역전 방지: min > max 케이스 처리
            if ratio_min > ratio_max:
                ratio_min, ratio_max = 0.2, 0.4  # fallback to safe default

            dynamic["m_freeze_ratio"] = (ratio_min, ratio_max)
            del dynamic["m_freeze"]
            reparam["m_freeze"] = "m_freeze_ratio * box_L"

        return dynamic, reparam

    def _get_default_value(self, spec: Tuple) -> Any:
        """스펙에서 기본값 추출 (중앙값)"""
        return self._get_median_value(spec)

    def _get_median_value(self, spec: Tuple) -> Any:
        """스펙에서 중앙값 계산"""
        if len(spec) >= 3 and spec[-1] == 'choice':
            options = spec[:-1]
            return options[len(options) // 2]
        elif len(spec) >= 3 and spec[-1] == 'int':
            return int((spec[0] + spec[1]) // 2)
        else:
            return (spec[0] + spec[1]) / 2


# ============================================================
# Weekly Auto-Freeze Pipeline
# ============================================================

def auto_reduce_for_next_week(
    current_space: Dict[str, Tuple],
    sensitivity_json_path: str,
    output_path: str = None
) -> ReducedParamSpace:
    """
    주간 자동 축소: 이번 주 sensitivity 결과로 다음 주 param_space 생성

    Args:
        current_space: 현재 param_space
        sensitivity_json_path: sensitivity 분석 결과 JSON
        output_path: 결과 저장 경로

    Returns:
        ReducedParamSpace
    """
    # Sensitivity 결과 로드
    with open(sensitivity_json_path, "r", encoding="utf-8") as f:
        sens_data = json.load(f)

    # Mock SensitivityResult 생성
    class MockSensitivityResult:
        def __init__(self, data):
            self.param_sensitivities = {}

            details = data.get("param_details", {})
            for param, info in details.items():
                class MockParamSens:
                    pass
                ps = MockParamSens()
                ps.sensitivity = info.get("sensitivity", 0.0)
                self.param_sensitivities[param] = ps

    mock_result = MockSensitivityResult(sens_data)

    # Reducer 실행
    reducer = ParamReducer(ReductionConfig(
        max_dynamic_params=5,  # 차원 축소 목표
        freeze_sensitivity_threshold=0.35,
    ))

    reduced = reducer.reduce_from_sensitivity(
        full_space=current_space,
        sensitivity_result=mock_result
    )

    # 결과 저장
    if output_path:
        output = {
            "timestamp": datetime.now().isoformat(),
            "source_sensitivity": sensitivity_json_path,
            **reduced.to_dict()
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"[ParamReducer] Saved reduced space to: {output_path}")
        print(f"  Original: {len(current_space)} params")
        print(f"  Reduced: {len(reduced.dynamic_params)} dynamic + {len(reduced.frozen_params)} frozen")
        print(f"  Reduction ratio: {reduced.reduction_ratio:.1%}")

    return reduced


__all__ = [
    "ParamReducer",
    "ReductionConfig",
    "ReducedParamSpace",
    "auto_reduce_for_next_week",
]
