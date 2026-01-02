"""
파라미터 스키마 정의 (Parameter Schema)
======================================

모든 튜닝 파라미터의 단일 정의 소스 (Single Source of Truth)

핵심 원칙:
1. 시간 관련 파라미터는 "분(minutes)" 단위로 저장
2. 런타임에서 타임프레임에 맞게 "봉(bars)"으로 변환
3. 비율 파라미터는 단위 무관

이점:
- 5m/15m/1h 어떤 타임프레임이든 동일한 의미
- 저장/비교/버전관리가 일관됨
- human-readable

Usage:
    from wpcn._08_tuning.param_schema import (
        TUNABLE_PARAMS_MINUTES,
        TUNABLE_PARAMS_RATIO,
        FIXED_PARAMS,
        minutes_to_bars,
        bars_to_minutes,
        convert_params_to_bars
    )

    # 분 단위 → 봉 단위 변환
    box_L = minutes_to_bars(1440, "15m")  # 1440분 → 96봉

    # 전체 파라미터 변환
    params_minutes = {"box_window_minutes": 1440, "freeze_minutes": 360}
    params_bars = convert_params_to_bars(params_minutes, "15m")
    # → {"box_L": 96, "m_freeze": 24}
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


# ============================================================
# 타임프레임 상수
# ============================================================

TF_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}


# ============================================================
# 시간 기반 튜닝 파라미터 (저장: 분 단위)
# ============================================================

TUNABLE_PARAMS_MINUTES = {
    # 박스/피봇 파라미터
    "box_window_minutes": (720, 2880),       # 12~48시간 (box_L)
    "freeze_minutes": (120, 480),            # 2~8시간 (m_freeze) - box의 1/6~1/3
    "pivot_window_minutes": (30, 120),       # 30분~2시간 (pivot_lr * 2 * tf_minutes)

    # 진입/청산 파라미터
    "cooldown_minutes": (60, 240),           # 1~4시간 (cooldown_bars)
    "max_hold_minutes": (720, 2880),         # 12~48시간 (max_hold_bars)

    # Reclaim/Fill 파라미터
    "reclaim_window_minutes": (15, 60),      # 15분~1시간 (N_reclaim)
    "fill_window_minutes": (15, 60),         # 15분~1시간 (N_fill)
}

# 분 → 봉 변환 시 키 매핑
MINUTES_TO_BARS_KEY_MAP = {
    "box_window_minutes": "box_L",
    "freeze_minutes": "m_freeze",
    "pivot_window_minutes": "_pivot_bars",   # pivot_lr = _pivot_bars // 2
    "cooldown_minutes": "cooldown_bars",
    "max_hold_minutes": "max_hold_bars",
    "reclaim_window_minutes": "N_reclaim",
    "fill_window_minutes": "N_fill",
}


# ============================================================
# 비율 기반 튜닝 파라미터 (단위 무관)
# ============================================================

TUNABLE_PARAMS_RATIO = {
    # ATR 관련
    "x_atr": (0.2, 0.5),                     # 돌파 임계값 (ATR 배수)
    "m_bw": (0.05, 0.15),                    # 박스폭 비율 (reclaim 레벨)

    # TP/SL
    "tp_pct": (0.01, 0.03),                  # 익절 비율 (1~3%)
    "sl_pct": (0.005, 0.015),                # 손절 비율 (0.5~1.5%)

    # 신호 점수
    "min_score": (3.0, 6.0),                 # 최소 신호 점수

    # Fill Probability
    "F_min": (0.2, 0.5),                     # 최소 체결 확률 (튜닝 시)
}


# ============================================================
# 고정 파라미터 (튜닝 금지)
# ============================================================

FIXED_PARAMS = {
    # Gate 철학 (쓰레기 필터링 - 건드리면 전략 성격 바뀜)
    "conf_min": 0.65,                        # 최소 신뢰도
    "edge_min": 0.70,                        # 최소 엣지
    "confirm_bars": 2,                       # 확인 봉 수
    "reclaim_hold_bars": 2,                  # reclaim 후 유지 확인

    # ATR 기간 (민감 - 튜닝하면 시장 바뀔 때 깨짐)
    "atr_len": 14,

    # CHAOS 레짐 판정
    "chaos_ret_atr": 3.0,

    # ADX 설정
    "adx_len": 14,
    "adx_trend": 20.0,
}

# 실행 현실성 파라미터 (별도 관리)
EXECUTION_PARAMS = {
    "N_fill": 3,                             # 주문 체결 허용 기간
    "F_min": 0.3,                            # 최소 체결 확률 (기본값)
}


# ============================================================
# 변환 유틸 함수
# ============================================================

def minutes_to_bars(minutes: int, timeframe: str) -> int:
    """
    분 → 봉 수 변환

    Args:
        minutes: 시간 (분)
        timeframe: 타임프레임 (예: "15m", "1h")

    Returns:
        봉 수 (최소 1)

    Example:
        >>> minutes_to_bars(1440, "15m")  # 24시간
        96
        >>> minutes_to_bars(120, "5m")    # 2시간
        24
    """
    tf_min = TF_MINUTES.get(timeframe, 15)
    return max(1, minutes // tf_min)


def bars_to_minutes(bars: int, timeframe: str) -> int:
    """
    봉 수 → 분 변환

    Args:
        bars: 봉 수
        timeframe: 타임프레임

    Returns:
        시간 (분)

    Example:
        >>> bars_to_minutes(96, "15m")
        1440
    """
    tf_min = TF_MINUTES.get(timeframe, 15)
    return bars * tf_min


def convert_params_to_bars(
    params_minutes: Dict[str, Any],
    timeframe: str
) -> Dict[str, Any]:
    """
    분 단위 파라미터를 봉 단위로 변환

    Args:
        params_minutes: 분 단위 파라미터 딕셔너리
        timeframe: 타임프레임

    Returns:
        봉 단위 파라미터 딕셔너리

    Example:
        >>> params = {"box_window_minutes": 1440, "freeze_minutes": 360, "tp_pct": 0.015}
        >>> convert_params_to_bars(params, "15m")
        {"box_L": 96, "m_freeze": 24, "tp_pct": 0.015}
    """
    result = {}

    for key, value in params_minutes.items():
        if key in MINUTES_TO_BARS_KEY_MAP:
            # 분 → 봉 변환
            bars_key = MINUTES_TO_BARS_KEY_MAP[key]
            bars_value = minutes_to_bars(int(value), timeframe)

            # pivot_lr은 특수 처리 (pivot_window = pivot_lr * 2 * tf_minutes)
            if bars_key == "_pivot_bars":
                result["pivot_lr"] = max(1, bars_value // 2)
            else:
                result[bars_key] = bars_value
        else:
            # 비율 파라미터는 그대로
            result[key] = value

    return result


def convert_params_to_minutes(
    params_bars: Dict[str, Any],
    timeframe: str
) -> Dict[str, Any]:
    """
    봉 단위 파라미터를 분 단위로 변환 (저장용)

    Args:
        params_bars: 봉 단위 파라미터 딕셔너리
        timeframe: 타임프레임

    Returns:
        분 단위 파라미터 딕셔너리
    """
    # 역방향 매핑 생성
    bars_to_minutes_map = {v: k for k, v in MINUTES_TO_BARS_KEY_MAP.items()}

    result = {}

    for key, value in params_bars.items():
        if key in bars_to_minutes_map:
            # 봉 → 분 변환
            minutes_key = bars_to_minutes_map[key]
            result[minutes_key] = bars_to_minutes(int(value), timeframe)
        elif key == "pivot_lr":
            # pivot_lr 특수 처리
            result["pivot_window_minutes"] = bars_to_minutes(int(value) * 2, timeframe)
        else:
            # 비율 파라미터는 그대로
            result[key] = value

    return result


# ============================================================
# 파라미터 검증
# ============================================================

def validate_tunable_params(params: Dict[str, Any]) -> Tuple[bool, str]:
    """
    튜닝 파라미터 범위 검증

    Args:
        params: 파라미터 딕셔너리 (분 또는 봉 단위)

    Returns:
        (is_valid, reason)
    """
    # 시간 기반 파라미터 검증
    for key, (min_val, max_val) in TUNABLE_PARAMS_MINUTES.items():
        if key in params:
            val = params[key]
            if not (min_val <= val <= max_val):
                return False, f"{key}={val} is out of range [{min_val}, {max_val}]"

    # 비율 기반 파라미터 검증
    for key, (min_val, max_val) in TUNABLE_PARAMS_RATIO.items():
        if key in params:
            val = params[key]
            if not (min_val <= val <= max_val):
                return False, f"{key}={val} is out of range [{min_val}, {max_val}]"

    # 종속 제약: freeze < box * 0.6
    box_minutes = params.get("box_window_minutes")
    freeze_minutes = params.get("freeze_minutes")
    if box_minutes and freeze_minutes:
        if freeze_minutes >= box_minutes * 0.6:
            return False, f"freeze_minutes({freeze_minutes}) >= box_window_minutes*0.6({box_minutes*0.6:.0f})"

    return True, "OK"


def is_fixed_param(key: str) -> bool:
    """해당 키가 고정 파라미터인지 확인"""
    return key in FIXED_PARAMS or key in EXECUTION_PARAMS


def get_all_tunable_keys() -> set:
    """모든 튜닝 가능한 키 반환"""
    return set(TUNABLE_PARAMS_MINUTES.keys()) | set(TUNABLE_PARAMS_RATIO.keys())


# ============================================================
# 기본값 생성
# ============================================================

@dataclass
class DefaultParams:
    """기본 파라미터 (분 단위)"""
    # 시간 기반 (중앙값)
    box_window_minutes: int = 1440           # 24시간
    freeze_minutes: int = 240                # 4시간
    pivot_window_minutes: int = 60           # 1시간
    cooldown_minutes: int = 120              # 2시간
    max_hold_minutes: int = 1440             # 24시간
    reclaim_window_minutes: int = 45         # 45분
    fill_window_minutes: int = 45            # 45분

    # 비율 기반 (중앙값)
    x_atr: float = 0.35
    m_bw: float = 0.10
    tp_pct: float = 0.02
    sl_pct: float = 0.01
    min_score: float = 4.5
    F_min: float = 0.3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box_window_minutes": self.box_window_minutes,
            "freeze_minutes": self.freeze_minutes,
            "pivot_window_minutes": self.pivot_window_minutes,
            "cooldown_minutes": self.cooldown_minutes,
            "max_hold_minutes": self.max_hold_minutes,
            "reclaim_window_minutes": self.reclaim_window_minutes,
            "fill_window_minutes": self.fill_window_minutes,
            "x_atr": self.x_atr,
            "m_bw": self.m_bw,
            "tp_pct": self.tp_pct,
            "sl_pct": self.sl_pct,
            "min_score": self.min_score,
            "F_min": self.F_min,
        }

    def to_bars(self, timeframe: str) -> Dict[str, Any]:
        """봉 단위로 변환"""
        return convert_params_to_bars(self.to_dict(), timeframe)


# 기본 인스턴스
DEFAULT_PARAMS = DefaultParams()
