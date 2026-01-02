"""
Adaptive Parameter Space Generator (v2 - GPT 지뢰 5개 반영)
============================================================

analyze_wyckoff_cycles_v2 분석 결과를 기반으로
Walk-Forward 최적화용 param_space를 동적으로 생성합니다.

GPT 지적 반영:
1. _hints는 param_space 밖으로 분리 → (param_space, hints) 튜플 반환
2. m_freeze <= box_L 종속 제약 → validate_params() 추가
3. params → Theta/RunConfig 어댑터 단일화 → build_theta_and_config()
4. complete + high_purity 기준만 사용 → 필터링 강화
5. optimizer에서 _ 시작 키 무시하도록 → 별도 처리 불필요 (분리됨)

흐름:
1. v2 분석 실행 (또는 JSON 로드)
2. complete + high_purity 사이클만 필터링
3. 통계 기반으로 (param_space, hints) 생성
4. WalkForwardOptimizer에 param_space만 전달
5. hints는 초기점/로깅용으로만 사용
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import numpy as np

# 기존 V8 파라미터 공간 (폴백용)
from .walk_forward import V8_PARAM_SPACE


# ============================================================
# GPT 지적 #3: params → Theta/RunConfig 어댑터 단일화
# ============================================================

# Theta 키 (박스/피봇 파라미터)
THETA_KEYS = {
    "pivot_lr", "box_L", "m_freeze", "atr_len",
    "x_atr", "m_bw", "N_reclaim", "N_fill", "F_min"
}

# RunConfig/BacktestConfig 키 (진입/청산/기타)
CONFIG_KEYS = {
    "tp_pct", "sl_pct", "min_score", "max_hold_bars",
    "rsi_oversold", "rsi_overbought", "cooldown_bars"
}


def build_theta_and_config(params: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """
    params dict를 Theta용과 RunConfig용으로 분리

    GPT 지적 #3 해결: 모듈마다 키 이름 해석 다른 문제 방지

    Args:
        params: 최적화에서 나온 파라미터 딕셔너리

    Returns:
        (theta_dict, config_dict)

    Example:
        >>> theta_dict, config_dict = build_theta_and_config(params)
        >>> theta = Theta(**theta_dict)
        >>> bt_config = BacktestConfig(**config_dict)
    """
    theta_dict = {k: v for k, v in params.items() if k in THETA_KEYS}
    config_dict = {k: v for k, v in params.items() if k in CONFIG_KEYS}

    return theta_dict, config_dict


# ============================================================
# GPT 지적 #2: m_freeze <= box_L 제약 검증
# ============================================================

def validate_params(params: Dict[str, Any]) -> Tuple[bool, str]:
    """
    파라미터 조합 유효성 검증

    GPT 지적 #2: m_freeze >= box_L 같은 말도 안되는 조합 차단

    Args:
        params: 파라미터 딕셔너리

    Returns:
        (is_valid, reason)
    """
    box_L = params.get("box_L", 96)
    m_freeze = params.get("m_freeze", 32)

    # m_freeze는 box_L의 절반 이하여야 함
    if m_freeze >= box_L:
        return False, f"m_freeze({m_freeze}) >= box_L({box_L})"

    if m_freeze > box_L * 0.6:
        return False, f"m_freeze({m_freeze}) > box_L*0.6({box_L*0.6:.0f})"

    # pivot_lr은 box_L/10 이하
    pivot_lr = params.get("pivot_lr", 4)
    if pivot_lr > box_L // 10:
        return False, f"pivot_lr({pivot_lr}) > box_L/10({box_L//10})"

    # TP/SL 비율 체크
    tp_pct = params.get("tp_pct", 0.015)
    sl_pct = params.get("sl_pct", 0.010)
    if tp_pct < sl_pct:
        return False, f"tp_pct({tp_pct}) < sl_pct({sl_pct})"

    return True, "ok"


def create_constrained_sampler(param_space: Dict[str, Any], rng=None):
    """
    제약 조건을 만족하는 샘플러 생성

    box_L을 먼저 샘플링하고, m_freeze는 그에 맞춰 샘플링

    Args:
        param_space: 파라미터 공간
        rng: numpy random generator

    Returns:
        sampler function
    """
    if rng is None:
        rng = np.random.default_rng(42)

    def sample_with_constraints() -> Dict[str, Any]:
        """제약 조건을 만족하는 파라미터 샘플링"""
        params = {}

        # 1. box_L 먼저 샘플링
        box_spec = param_space.get("box_L", (48, 192, 'int'))
        if len(box_spec) == 3 and box_spec[2] == 'int':
            params["box_L"] = int(rng.integers(box_spec[0], box_spec[1] + 1))
        else:
            params["box_L"] = int(rng.uniform(box_spec[0], box_spec[1]))

        box_L = params["box_L"]

        # 2. m_freeze는 box_L 기준으로 샘플링 (box_L의 1/6 ~ 1/2)
        m_freeze_spec = param_space.get("m_freeze", (16, 64, 'int'))
        m_freeze_min = max(m_freeze_spec[0], box_L // 6)
        m_freeze_max = min(m_freeze_spec[1], box_L // 2)
        if m_freeze_max <= m_freeze_min:
            m_freeze_max = m_freeze_min + 1
        params["m_freeze"] = int(rng.integers(m_freeze_min, m_freeze_max + 1))

        # 3. pivot_lr은 box_L/10 이하
        pivot_spec = param_space.get("pivot_lr", (2, 8, 'int'))
        pivot_max = min(pivot_spec[1], box_L // 10)
        pivot_min = min(pivot_spec[0], pivot_max)
        params["pivot_lr"] = int(rng.integers(pivot_min, pivot_max + 1))

        # 4. 나머지 파라미터 독립 샘플링
        for key, spec in param_space.items():
            if key in params:  # 이미 처리됨
                continue

            if isinstance(spec, tuple):
                if len(spec) == 3 and spec[2] == 'int':
                    params[key] = int(rng.integers(spec[0], spec[1] + 1))
                elif len(spec) == 2:
                    params[key] = float(rng.uniform(spec[0], spec[1]))
                elif spec[-1] == 'choice':
                    params[key] = rng.choice(spec[:-1])
            elif isinstance(spec, list):
                params[key] = rng.choice(spec)

        return params

    return sample_with_constraints


# ============================================================
# GPT 지적 #1: hints 분리
# ============================================================

@dataclass
class ParamHints:
    """
    최적화 힌트 (param_space와 분리)

    Bayesian 초기점이나 로깅용으로만 사용
    param_space에 절대 포함시키지 않음!
    """
    recommended_box_L: Optional[int] = None
    recommended_m_freeze: Optional[int] = None
    recommended_max_hold: Optional[int] = None
    fft_dominant_cycle: Optional[int] = None
    fft_reliable: bool = False
    median_cycle: Optional[float] = None
    confidence: str = "low"
    source: str = "default"

    def to_initial_point(self, param_space: Dict) -> Optional[Dict]:
        """
        Bayesian 최적화 초기점 생성

        Returns:
            초기점 dict (param_space 키와 일치)
        """
        if self.confidence == "low":
            return None

        x0 = {}
        if self.recommended_box_L:
            x0["box_L"] = self.recommended_box_L
        if self.recommended_m_freeze:
            x0["m_freeze"] = self.recommended_m_freeze

        return x0 if x0 else None


@dataclass
class AdaptiveParamSpace:
    """
    분석 결과 기반 적응형 파라미터 공간

    GPT 지적 반영:
    - hints는 별도 객체로 분리
    - complete + high_purity 사이클만 사용
    """
    # Wyckoff Box 파라미터
    box_L: List[int] = field(default_factory=lambda: [48, 72, 96, 120, 144, 192])
    m_freeze: List[int] = field(default_factory=lambda: [16, 24, 32, 48, 64])
    pivot_lr: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 8])
    atr_len: List[int] = field(default_factory=lambda: [10, 14, 20])

    # 실수형 파라미터 (min, max)
    x_atr: Tuple[float, float] = (0.2, 0.5)
    m_bw: Tuple[float, float] = (0.05, 0.15)

    # 진입/청산 파라미터
    tp_pct: Tuple[float, float] = (0.01, 0.03)
    sl_pct: Tuple[float, float] = (0.005, 0.015)
    min_score: Tuple[float, float] = (3.0, 5.0)

    # RSI 파라미터
    rsi_oversold: List[int] = field(default_factory=lambda: [25, 30, 35, 40])
    rsi_overbought: List[int] = field(default_factory=lambda: [60, 65, 70, 75])

    # 메타 정보
    source: str = "default"
    confidence: str = "low"
    analysis_date: Optional[str] = None

    def to_optimizer_space(self) -> Dict[str, Any]:
        """
        WalkForwardOptimizer 호환 형식으로 변환

        Returns:
            param_space (hints 미포함!)
        """
        return {
            "box_L": (min(self.box_L), max(self.box_L), 'int'),
            "m_freeze": (min(self.m_freeze), max(self.m_freeze), 'int'),
            "pivot_lr": (min(self.pivot_lr), max(self.pivot_lr), 'int'),
            "atr_len": (min(self.atr_len), max(self.atr_len), 'int'),
            "x_atr": self.x_atr,
            "m_bw": self.m_bw,
            "tp_pct": self.tp_pct,
            "sl_pct": self.sl_pct,
            "min_score": self.min_score,
            "rsi_oversold": (min(self.rsi_oversold), max(self.rsi_oversold), 'int'),
            "rsi_overbought": (min(self.rsi_overbought), max(self.rsi_overbought), 'int'),
        }

    def to_grid_space(self) -> Dict[str, List]:
        """GridSearchOptimizer 호환 형식"""
        def linspace_list(min_v: float, max_v: float, n: int = 5) -> List[float]:
            return [round(v, 4) for v in np.linspace(min_v, max_v, n)]

        return {
            "box_L": self.box_L,
            "m_freeze": self.m_freeze,
            "pivot_lr": self.pivot_lr,
            "atr_len": self.atr_len,
            "x_atr": linspace_list(self.x_atr[0], self.x_atr[1], 4),
            "m_bw": linspace_list(self.m_bw[0], self.m_bw[1], 4),
            "tp_pct": linspace_list(self.tp_pct[0], self.tp_pct[1], 4),
            "sl_pct": linspace_list(self.sl_pct[0], self.sl_pct[1], 4),
            "min_score": linspace_list(self.min_score[0], self.min_score[1], 3),
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
        }


# ============================================================
# GPT 지적 #4: complete + high_purity만 사용
# ============================================================

def generate_adaptive_space(
    analysis_result: Dict[str, Any],
    timeframe: str = "15m",
    confidence_threshold: str = "medium",
    purity_threshold: float = 0.6
) -> Tuple[AdaptiveParamSpace, ParamHints]:
    """
    v2 분석 결과에서 (AdaptiveParamSpace, ParamHints) 생성

    GPT 지적 반영:
    - hints는 param_space와 분리하여 반환
    - complete + high_purity 사이클만 기준으로 통계 계산

    Args:
        analysis_result: analyze_wyckoff_cycles_v2.py 출력 JSON
        timeframe: 기준 타임프레임
        confidence_threshold: 최소 신뢰도
        purity_threshold: 최소 purity (기본 0.6)

    Returns:
        (param_space, hints) 튜플
    """
    space = AdaptiveParamSpace()
    hints = ParamHints()

    # 타임프레임 데이터 추출
    tf_data = analysis_result.get("timeframes", {}).get(timeframe, {})
    if not tf_data:
        print(f"[AdaptiveSpace] No data for {timeframe}, using defaults")
        space.source = "default_no_data"
        hints.source = "default_no_data"
        return space, hints

    wyckoff_stats = tf_data.get("wyckoff_stats", {})
    fft_cycles = tf_data.get("fft_cycles", {})
    reco = tf_data.get("recommended_params", {})
    overall = wyckoff_stats.get("overall", {})

    # 신뢰도 체크
    confidence = reco.get("confidence", "low")
    confidence_order = {"low": 0, "medium": 1, "high": 2}
    threshold_level = confidence_order.get(confidence_threshold, 1)
    current_level = confidence_order.get(confidence, 0)

    if current_level < threshold_level:
        print(f"[AdaptiveSpace] Confidence {confidence} < {confidence_threshold}")
        space.source = f"low_confidence_{confidence}"
        hints.source = f"low_confidence_{confidence}"
        hints.confidence = confidence
        return space, hints

    # ============================================================
    # GPT 지적 #4 핵심: complete + high_purity만 사용
    # ============================================================
    # cycles_detail에서 직접 필터링
    cycles_detail = tf_data.get("cycles_detail", [])

    # complete + high_purity 사이클만 필터
    quality_cycles = [
        c for c in cycles_detail
        if c.get("complete", False)
        and c.get("direction_purity", 0) >= purity_threshold
    ]

    if len(quality_cycles) < 3:
        print(f"[AdaptiveSpace] Too few quality cycles ({len(quality_cycles)}), using overall stats")
        # 폴백: 전체 complete cycles 사용
        quality_cycles = [c for c in cycles_detail if c.get("complete", False)]

    if quality_cycles:
        # quality cycles 기반 통계 재계산
        lengths = [c.get("length", 0) for c in quality_cycles if c.get("length", 0) > 0]
        if lengths:
            median_len = float(np.median(lengths))
            p25 = float(np.percentile(lengths, 25))
            p75 = float(np.percentile(lengths, 75))
            print(f"[AdaptiveSpace] Using {len(lengths)} quality cycles: "
                  f"median={median_len:.0f}, p25={p25:.0f}, p75={p75:.0f}")
        else:
            median_len = overall.get("median_length", 96)
            p25 = overall.get("percentile_25", 72)
            p75 = overall.get("percentile_75", 144)
    else:
        median_len = overall.get("median_length", 96)
        p25 = overall.get("percentile_25", 72)
        p75 = overall.get("percentile_75", 144)

    # ============================================================
    # box_L 탐색 범위 생성
    # ============================================================
    fft_dominant = fft_cycles.get("dominant_cycle", 0)
    fft_reliable = fft_cycles.get("reliable", False)

    if median_len > 0:
        min_box = max(24, int(p25 * 0.8))
        max_box = min(300, int(p75 * 1.2))

        box_L_candidates = list(set([
            min_box,
            int(p25),
            int(median_len * 0.9),
            int(median_len),
            int(median_len * 1.1),
            int(p75),
            max_box
        ]))

        if fft_reliable and fft_dominant > 0:
            if min_box <= fft_dominant <= max_box:
                box_L_candidates.append(int(fft_dominant))

        box_L_candidates = sorted(set([b for b in box_L_candidates if 24 <= b <= 300]))
        space.box_L = box_L_candidates

    # ============================================================
    # m_freeze 탐색 범위 (box_L 종속)
    # ============================================================
    if space.box_L:
        # GPT 지적 #2: box_L의 1/6 ~ 1/2 범위
        min_freeze = max(8, min(space.box_L) // 6)
        max_freeze = max(space.box_L) // 2
        space.m_freeze = sorted(set([
            min_freeze,
            min_freeze + (max_freeze - min_freeze) // 4,
            (min_freeze + max_freeze) // 2,
            min_freeze + 3 * (max_freeze - min_freeze) // 4,
            max_freeze
        ]))

    # ============================================================
    # hints 설정 (param_space와 분리!)
    # ============================================================
    hints.recommended_box_L = reco.get("recommended_box_L")
    hints.recommended_m_freeze = int(median_len // 3) if median_len else None
    hints.recommended_max_hold = reco.get("recommended_max_hold_bars")
    hints.fft_dominant_cycle = fft_dominant if fft_dominant else None
    hints.fft_reliable = fft_reliable
    hints.median_cycle = median_len
    hints.confidence = confidence
    hints.source = f"v2_analysis_{timeframe}"

    # 메타 정보
    space.source = f"v2_analysis_{timeframe}"
    space.confidence = confidence
    space.analysis_date = analysis_result.get("analysis_date")

    print(f"[AdaptiveSpace] Generated from {timeframe} analysis")
    print(f"  box_L range: {space.box_L}")
    print(f"  m_freeze range: {space.m_freeze}")
    print(f"  confidence: {confidence}")
    print(f"  hints: box_L={hints.recommended_box_L}, fft={hints.fft_dominant_cycle}")

    return space, hints


def load_analysis_and_generate_space(
    json_path: str = "wyckoff_cycle_analysis_v2.json",
    timeframe: str = "15m"
) -> Tuple[AdaptiveParamSpace, ParamHints]:
    """JSON 파일에서 로드 후 (param_space, hints) 생성"""
    path = Path(json_path)
    if not path.exists():
        print(f"[AdaptiveSpace] {json_path} not found, returning defaults")
        return AdaptiveParamSpace(source="default_file_not_found"), ParamHints()

    with open(path, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    return generate_adaptive_space(analysis, timeframe)


def get_adaptive_param_space(
    analysis_json_path: str = None,
    timeframe: str = "15m",
    fallback_to_defaults: bool = True
) -> Tuple[Dict[str, Any], ParamHints]:
    """
    분석 결과 기반 (param_space, hints) 반환

    GPT 지적 반영: hints는 분리하여 반환

    Returns:
        (param_space, hints) 튜플
    """
    default_paths = [
        "wyckoff_cycle_analysis_v2.json",
        "results/wyckoff_cycle_analysis_v2.json",
        "../wyckoff_cycle_analysis_v2.json"
    ]

    json_path = analysis_json_path
    if json_path is None:
        for path in default_paths:
            if Path(path).exists():
                json_path = path
                break

    if json_path and Path(json_path).exists():
        space, hints = load_analysis_and_generate_space(json_path, timeframe)
        param_space = space.to_optimizer_space()
        return param_space, hints
    elif fallback_to_defaults:
        print("[AdaptiveSpace] No analysis found, using V8_PARAM_SPACE defaults")
        return V8_PARAM_SPACE.copy(), ParamHints()
    else:
        raise FileNotFoundError(f"Analysis file not found: {analysis_json_path}")


__all__ = [
    # Core
    "AdaptiveParamSpace",
    "ParamHints",
    "generate_adaptive_space",
    "load_analysis_and_generate_space",
    "get_adaptive_param_space",
    # Utilities (GPT 지적 반영)
    "build_theta_and_config",
    "validate_params",
    "create_constrained_sampler",
    "THETA_KEYS",
    "CONFIG_KEYS",
]
