"""
Barrier Hit Probability Model V2
배리어 도달 확률 모델 (GPT 피드백 반영)

FIXED (P2 보완):
- (A) Finite Horizon: 시간 제한(max_hold_bars) 내 확률 계산 (Monte Carlo)
- (B) 단위 통일: 로그 수익률 공간에서 계산
- (C) Drift 최소화: 지표 2개만 사용 + 클램프
- (D) 수치 안정: expm1 사용 + 하한/상한 클램프
- (E) EV 정의: R-unit 기반 + 비용 포함
- (F) 캘리브레이션: 3-class (TP/SL/timeout)

핵심 공식 (Drift-Diffusion Model with Time Limit):
Monte Carlo로 "T 시간 내 TP 먼저 도달할 확률" 계산
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ExitType(Enum):
    """청산 유형 (3-class for calibration)"""
    TP = "tp"
    SL = "sl"
    TIMEOUT = "timeout"


@dataclass
class BarrierProbability:
    """
    배리어 도달 확률 결과

    FIXED (E): EV는 R-unit 기반으로 계산
    - R = TP거리 / SL거리 (risk unit)
    - EV_R = p_tp * R - p_sl * 1 - costs_R

    Attributes:
        p_tp: TP 도달 확률 (0~1)
        p_sl: SL 도달 확률 (0~1)
        p_timeout: 시간초과 확률 (0~1)
        ev_r: 기대값 (R-unit 기준, 비용 포함)
        ev_pct: 기대 수익률 (%)
        drift: 추정된 drift (로그 수익률/봉)
        volatility: 추정된 volatility (로그 수익률 std/봉)
        r_ratio: Risk-Reward 비율
        confidence: 추정 신뢰도 (0~1)
    """
    p_tp: float
    p_sl: float
    p_timeout: float
    ev_r: float
    ev_pct: float
    drift: float
    volatility: float
    r_ratio: float
    confidence: float = 0.5

    @property
    def edge(self) -> float:
        """기대 엣지: EV가 양수면 edge 있음"""
        return self.ev_r

    @property
    def is_favorable(self) -> bool:
        """유리한 확률인지"""
        return self.ev_r > 0

    def to_dict(self) -> Dict[str, float]:
        return {
            'p_tp': self.p_tp,
            'p_sl': self.p_sl,
            'p_timeout': self.p_timeout,
            'ev_r': self.ev_r,
            'ev_pct': self.ev_pct,
            'drift': self.drift,
            'volatility': self.volatility,
            'r_ratio': self.r_ratio,
            'confidence': self.confidence,
            'edge': self.edge
        }


# ============================================================
# FIXED (A): Finite Horizon Monte Carlo
# ============================================================

def monte_carlo_barrier_probability(
    drift: float,
    volatility: float,
    tp_distance: float,
    sl_distance: float,
    max_bars: int,
    n_simulations: int = 5000,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Monte Carlo로 시간 제한 내 배리어 도달 확률 계산

    FIXED (A): 무한 시간 공식 대신 finite horizon 사용
    FIXED (P0-1): seed 필수로 재현성 확보
    FIXED (P2-7): 벡터화로 성능 개선

    Args:
        drift: 봉당 기대 로그수익률 (μ)
        volatility: 봉당 로그수익률 표준편차 (σ)
        tp_distance: TP까지 로그 거리 (양수)
        sl_distance: SL까지 로그 거리 (양수, 아래 방향이므로 절대값)
        max_bars: 최대 보유 봉 수
        n_simulations: 시뮬레이션 횟수
        seed: 난수 시드 (재현성용, 필수 권장!)

    Returns:
        (p_tp, p_sl, p_timeout): 확률 튜플
    """
    # FIXED (P0-1): seed 설정 (재현성)
    rng = np.random.default_rng(seed)

    # FIXED (D): 수치 안정성
    volatility = max(volatility, 1e-6)
    tp_distance = max(tp_distance, 1e-8)
    sl_distance = max(sl_distance, 1e-8)

    # 벡터화된 시뮬레이션 (성능 최적화)
    random_walks = rng.normal(
        loc=drift,
        scale=volatility,
        size=(n_simulations, max_bars)
    )

    # 누적 경로
    cumulative_paths = np.cumsum(random_walks, axis=1)

    # FIXED (P2-7): 완전 벡터화로 성능 개선
    # TP 도달: 경로가 tp_distance 이상인 첫 번째 인덱스
    tp_hit_mask = cumulative_paths >= tp_distance
    # SL 도달: 경로가 -sl_distance 이하인 첫 번째 인덱스
    sl_hit_mask = cumulative_paths <= -sl_distance

    # 각 경로별 첫 hit 인덱스 (없으면 max_bars)
    # argmax는 첫 번째 True 위치 반환, 모두 False면 0 반환
    tp_first_hit = np.where(
        tp_hit_mask.any(axis=1),
        tp_hit_mask.argmax(axis=1),
        max_bars  # 도달 안함
    )
    sl_first_hit = np.where(
        sl_hit_mask.any(axis=1),
        sl_hit_mask.argmax(axis=1),
        max_bars  # 도달 안함
    )

    # 결과 판정 (벡터화)
    # TP 먼저: tp_first_hit < sl_first_hit AND tp_first_hit < max_bars
    tp_wins = (tp_first_hit < sl_first_hit) & (tp_first_hit < max_bars)
    # SL 먼저: sl_first_hit < tp_first_hit AND sl_first_hit < max_bars
    sl_wins = (sl_first_hit < tp_first_hit) & (sl_first_hit < max_bars)
    # 동시 도달 (같은 봉): SL 처리 (보수적)
    same_bar = (tp_first_hit == sl_first_hit) & (tp_first_hit < max_bars)
    sl_wins = sl_wins | same_bar
    # Timeout: 둘 다 도달 안함
    timeout = (tp_first_hit >= max_bars) & (sl_first_hit >= max_bars)

    # 확률 계산
    p_tp = tp_wins.sum() / n_simulations
    p_sl = sl_wins.sum() / n_simulations
    p_timeout = timeout.sum() / n_simulations

    # 정규화 (합이 1이 되도록)
    total_prob = p_tp + p_sl + p_timeout
    if total_prob > 0:
        p_tp /= total_prob
        p_sl /= total_prob
        p_timeout /= total_prob

    # FIXED (P0-8): 극단값 방지 + 재정규화
    # 세 확률 모두에 eps 적용 후 재정규화해서 합이 1 유지
    eps = 0.001
    p_tp = max(p_tp, eps)
    p_sl = max(p_sl, eps)
    p_timeout = max(p_timeout, 0.0)  # timeout은 0이어도 됨

    # 재정규화 (클램프 후 합이 1 초과할 수 있으므로)
    total_after_clamp = p_tp + p_sl + p_timeout
    if total_after_clamp > 0:
        p_tp /= total_after_clamp
        p_sl /= total_after_clamp
        p_timeout /= total_after_clamp

    return float(p_tp), float(p_sl), float(p_timeout)


def first_passage_probability_analytic(
    drift: float,
    volatility: float,
    tp_distance: float,
    sl_distance: float
) -> Tuple[float, float]:
    """
    Analytic First-Passage Probability (무한 시간, 참고용)

    FIXED (D): expm1 사용 + 수치 안정화

    Returns:
        (p_tp, p_sl): 확률 튜플 (timeout 없음)
    """
    # FIXED (D): 수치 안정성
    sigma_sq = max(volatility ** 2, 1e-12)
    tp_distance = max(tp_distance, 1e-8)
    sl_distance = max(sl_distance, 1e-8)

    # Drift가 0에 가까운 경우: 순수 랜덤워크
    if abs(drift) < 1e-10:
        # P(TP first) = sl_distance / (tp_distance + sl_distance)
        p_tp = sl_distance / (tp_distance + sl_distance)
        p_sl = 1.0 - p_tp
        return p_tp, p_sl

    # First-Passage Probability 공식
    # P(τ_U < τ_L) = [1 - exp(-2μ*sl_dist/σ²)] / [1 - exp(-2μ*(tp_dist+sl_dist)/σ²)]

    # FIXED (D): expm1 사용으로 수치 안정화
    # expm1(x) = exp(x) - 1, x가 0 근처일 때 더 정확
    exp_num = -2 * drift * sl_distance / sigma_sq
    exp_denom = -2 * drift * (tp_distance + sl_distance) / sigma_sq

    # 오버플로우 방지
    exp_num = np.clip(exp_num, -500, 500)
    exp_denom = np.clip(exp_denom, -500, 500)

    try:
        # 1 - exp(x) = -expm1(x)
        numerator = -np.expm1(exp_num)
        denominator = -np.expm1(exp_denom)

        if abs(denominator) < 1e-10:
            p_tp = 0.5
        else:
            p_tp = numerator / denominator

        # 확률 범위 클리핑
        p_tp = np.clip(p_tp, 0.001, 0.999)
        p_sl = 1.0 - p_tp

        return p_tp, p_sl

    except (OverflowError, FloatingPointError) as e:
        logger.warning(f"Numerical error in analytic calc: {e}")
        return 0.5, 0.5


# ============================================================
# FIXED (B): 단위 통일 (로그 수익률 공간)
# ============================================================

def estimate_volatility_from_atr(
    atr: float,
    current_price: float
) -> float:
    """
    ATR로부터 volatility(σ) 추정

    FIXED (B): 로그 수익률 공간으로 통일
    - ATR → %ATR → 봉당 로그수익률 σ

    Args:
        atr: ATR 값 (가격 단위)
        current_price: 현재 가격

    Returns:
        volatility: 봉당 로그수익률 표준편차
    """
    if current_price <= 0 or atr <= 0:
        return 0.01  # 기본값

    # ATR을 %로 변환
    atr_pct = atr / current_price

    # ATR은 True Range의 평균 (High-Low 범위)
    # 일반적으로 σ ≈ ATR% / 1.5~2.0
    # 로그 수익률로 변환: log(1+x) ≈ x for small x
    bar_volatility = atr_pct / 1.6

    # FIXED (D): 하한 클램프
    return max(bar_volatility, 0.0001)


# ============================================================
# FIXED (P0-4): 가격 기반 모멘텀 추출
# ============================================================

def estimate_momentum_from_price(
    closes: np.ndarray,
    lookback: int = 20
) -> float:
    """
    가격 데이터에서 모멘텀 추정

    FIXED (P0-4): score에서 추출하지 않고 가격 직접 사용
    - 최근 N봉의 로그수익률 평균을 모멘텀으로 사용
    - 이는 drift 추정의 입력이 되어 순환 참조 방지

    Args:
        closes: 종가 배열 (최소 lookback+1개 필요)
        lookback: 모멘텀 계산 기간

    Returns:
        momentum: -1 ~ 1 범위의 정규화된 모멘텀
    """
    if len(closes) < lookback + 1:
        return 0.0

    # 로그 수익률 계산
    recent_closes = closes[-(lookback + 1):]
    log_returns = np.diff(np.log(recent_closes))

    if len(log_returns) == 0:
        return 0.0

    # 평균 수익률
    avg_return = np.mean(log_returns)

    # 표준편차 (정규화용)
    std_return = np.std(log_returns)
    if std_return < 1e-8:
        std_return = 0.01  # 최소값

    # z-score → -1 ~ 1 클리핑
    # 2 표준편차 = ±1
    momentum = np.clip(avg_return / (std_return * 2), -1.0, 1.0)

    return float(momentum)


# ============================================================
# FIXED (C): Drift 추정 최소화
# ============================================================

def estimate_drift_minimal(
    recent_returns: Optional[np.ndarray] = None,
    div_strength: float = 0.0,
    is_long: bool = True,
    lookback: int = 20
) -> float:
    """
    Drift(μ) 추정 - 최소주의 버전

    FIXED (C): 복잡한 지표 대신 2가지만 사용
    1. 최근 수익률 평균 (있으면)
    2. 다이버전스 강도

    Args:
        recent_returns: 최근 N봉의 로그수익률 배열
        div_strength: 다이버전스 강도 (-1 ~ 1)
        is_long: 롱 포지션 여부
        lookback: 수익률 평균 기간

    Returns:
        drift: 봉당 기대 로그수익률
    """
    base_drift = 0.0

    # 1. 최근 수익률 평균 (있으면)
    if recent_returns is not None and len(recent_returns) >= lookback:
        # 최근 lookback개 수익률의 평균
        base_drift = np.mean(recent_returns[-lookback:])

    # 2. 다이버전스 보정 (약하게)
    # FIXED (C): 보정 강도 제한 (±0.0005 = ±0.05%/봉)
    div_adjustment = np.clip(div_strength * 0.0005, -0.0005, 0.0005)

    # 롱/숏 방향 적용
    if not is_long:
        base_drift = -base_drift
        div_adjustment = -div_adjustment

    drift = base_drift + div_adjustment

    # FIXED (C): drift 클램프 (±0.002 = ±0.2%/봉)
    # 너무 큰 drift는 확률을 극단으로 밀어버림
    return np.clip(drift, -0.002, 0.002)


def estimate_drift_from_indicators(
    momentum: float = 0.0,
    rsi: float = 50.0,
    div_strength: float = 0.0,
    wyckoff_phase: str = "unknown",
    phase_confidence: float = 0.5,
    is_long: bool = True,
    use_mean_reversion: bool = True  # FIXED (P0-5): 평균회귀 모드
) -> float:
    """
    지표 기반 Drift 추정 (레거시 호환)

    FIXED (C): 모든 조정값에 강한 클램프 적용
    FIXED (P0-5): RSI 방향 수정 (평균회귀 전략 지원)

    Args:
        momentum: 모멘텀 (-1 ~ 1)
        rsi: RSI (0 ~ 100)
        div_strength: 다이버전스 강도
        wyckoff_phase: Wyckoff 페이즈
        phase_confidence: 페이즈 신뢰도
        is_long: 롱 포지션 여부
        use_mean_reversion: 평균회귀 모드 (RSI 방향 반전)

    Returns:
        drift: 봉당 기대 로그수익률
    """
    # 기본 drift는 0 (중립)
    drift = 0.0

    # 모멘텀 기여 (매우 약하게)
    # FIXED (C): 클램프 적용
    momentum_adj = np.clip(momentum * 0.0003, -0.0003, 0.0003)

    # FIXED (P0-5): RSI 기여 - 평균회귀 모드
    # 평균회귀 전략: RSI 높으면 하락 예상 (음수), RSI 낮으면 상승 예상 (양수)
    # 추세추종 전략: RSI 높으면 상승 모멘텀 (양수), RSI 낮으면 하락 모멘텀 (음수)
    rsi_deviation = (rsi - 50) / 50  # -1 ~ 1
    if use_mean_reversion:
        # 평균회귀: 부호 반전
        rsi_adj = np.clip(-rsi_deviation * 0.0002, -0.0002, 0.0002)
    else:
        # 추세추종: 원래 방향
        rsi_adj = np.clip(rsi_deviation * 0.0002, -0.0002, 0.0002)

    # 다이버전스 기여 (다이버전스는 항상 반전 신호)
    div_adj = np.clip(div_strength * 0.0004, -0.0004, 0.0004)

    # Wyckoff 페이즈 기여
    phase_adj = _get_wyckoff_drift_adjustment_v2(wyckoff_phase, phase_confidence)

    # 합산
    drift = momentum_adj + rsi_adj + div_adj + phase_adj

    # 롱/숏 방향
    if not is_long:
        drift = -drift

    # FIXED (C): 최종 클램프
    return np.clip(drift, -0.002, 0.002)


def _get_wyckoff_drift_adjustment_v2(
    phase: str,
    confidence: float
) -> float:
    """
    Wyckoff 페이즈 drift 조정 V2

    FIXED (C): 조정값 대폭 축소 (기존 0.3~0.5 → 0.0003~0.0005)
    """
    # 페이즈별 기본 조정값 (매우 작게!)
    phase_adjustments = {
        'accumulation': 0.0003,
        'distribution': -0.0003,
        'spring': 0.0005,
        'upthrust': -0.0005,
        'markup': 0.0002,
        'markdown': -0.0002,
        'ranging': 0.0,
        'unknown': 0.0
    }

    base_adj = phase_adjustments.get(phase.lower(), 0.0)

    # 신뢰도 적용 + 클램프
    adj = base_adj * np.clip(confidence, 0, 1)
    return np.clip(adj, -0.0005, 0.0005)


# ============================================================
# FIXED (E): EV 정의 (R-unit + 비용)
# ============================================================

def calculate_ev_with_costs(
    p_tp: float,
    p_sl: float,
    p_timeout: float,
    r_ratio: float,
    tp_pct: float,
    sl_pct: float,
    entry_fee_pct: float = 0.0004,
    exit_fee_pct: float = 0.0004,
    slippage_pct: float = 0.0001
) -> Tuple[float, float]:
    """
    비용 포함 기대값 계산

    FIXED (E): R-unit 기반 EV + 비용 명시적 반영
    FIXED (P0-3): ev_r에도 비용 반영

    Args:
        p_tp: TP 확률
        p_sl: SL 확률
        p_timeout: 시간초과 확률
        r_ratio: Risk-Reward 비율 (TP거리/SL거리)
        tp_pct: TP 수익률 (%)
        sl_pct: SL 손실률 (%)
        entry_fee_pct: 진입 수수료
        exit_fee_pct: 청산 수수료
        slippage_pct: 슬리피지

    Returns:
        (ev_r, ev_pct): R-unit EV (비용 포함), % EV (비용 포함)
    """
    # 총 비용 (왕복)
    total_cost_pct = entry_fee_pct + exit_fee_pct + slippage_pct * 2

    # FIXED (P0-3): R-unit 비용 계산
    # 비용을 R-unit으로 변환: cost_r = cost_pct / sl_pct
    # (SL 손실 = 1R이므로, sl_pct가 1R에 해당)
    cost_r = total_cost_pct / sl_pct if sl_pct > 0 else 0.0

    # R-unit EV (비용 포함)
    # TP 시: +R, SL 시: -1, Timeout 시: 0 (가정: 본전 청산)
    # 모든 경우에 비용 발생
    ev_r_gross = p_tp * r_ratio - p_sl * 1.0 + p_timeout * 0.0
    ev_r = ev_r_gross - cost_r

    # % EV (비용 포함)
    # TP 시: +tp_pct, SL 시: -sl_pct, Timeout 시: ~0
    ev_pct_gross = p_tp * tp_pct - p_sl * sl_pct
    ev_pct = ev_pct_gross - total_cost_pct

    return ev_r, ev_pct


# ============================================================
# 메인 함수
# ============================================================

def calculate_barrier_probability(
    current_price: float,
    tp_price: float,
    sl_price: float,
    atr: float,
    max_hold_bars: int = 48,
    momentum: float = 0.0,
    rsi: float = 50.0,
    div_strength: float = 0.0,
    wyckoff_phase: str = "unknown",
    phase_confidence: float = 0.5,
    is_long: bool = True,
    entry_fee_pct: float = 0.0004,
    exit_fee_pct: float = 0.0004,
    slippage_pct: float = 0.0001,
    use_monte_carlo: bool = True,
    n_simulations: int = 3000,
    use_mean_reversion: bool = True,  # FIXED (P0-5): 평균회귀 모드
    seed: Optional[int] = None  # FIXED (P0-7): MC 재현성을 위한 seed
) -> BarrierProbability:
    """
    종합 배리어 도달 확률 계산 V2

    FIXED 전체 반영:
    - (A) Monte Carlo로 finite horizon 확률
    - (B) 로그 수익률 공간에서 계산
    - (C) Drift 클램프
    - (D) 수치 안정화
    - (E) R-unit EV + 비용
    - (F) 3-class 확률 (TP/SL/timeout)
    - (P0-5) 평균회귀 전략 RSI 방향 수정
    - (P0-7) MC seed 전달로 재현성 확보

    Args:
        current_price: 현재 가격
        tp_price: TP 가격
        sl_price: SL 가격
        atr: ATR 값
        max_hold_bars: 최대 보유 봉 수 (시간 제한!)
        momentum: 모멘텀 (-1 ~ 1)
        rsi: RSI (0 ~ 100)
        div_strength: 다이버전스 강도
        wyckoff_phase: Wyckoff 페이즈
        phase_confidence: 페이즈 신뢰도
        is_long: 롱 포지션 여부
        entry_fee_pct: 진입 수수료
        exit_fee_pct: 청산 수수료
        slippage_pct: 슬리피지
        use_monte_carlo: MC 사용 여부
        n_simulations: MC 시뮬레이션 횟수
        use_mean_reversion: 평균회귀 전략 여부 (RSI 방향 반전)
        seed: MC 난수 시드 (재현성용, None이면 랜덤)

    Returns:
        BarrierProbability: 확률 결과
    """
    # === 입력 검증 ===
    if current_price <= 0:
        return _default_probability()

    # === FIXED (B): 로그 거리 계산 ===
    if is_long:
        tp_log_dist = np.log(tp_price / current_price)  # 양수
        sl_log_dist = np.log(current_price / sl_price)  # 양수 (절대값)
        tp_pct = (tp_price - current_price) / current_price
        sl_pct = (current_price - sl_price) / current_price
    else:
        tp_log_dist = np.log(current_price / tp_price)  # 양수
        sl_log_dist = np.log(sl_price / current_price)  # 양수
        tp_pct = (current_price - tp_price) / current_price
        sl_pct = (sl_price - current_price) / current_price

    # R-ratio
    r_ratio = tp_log_dist / sl_log_dist if sl_log_dist > 0 else 1.0

    # === FIXED (B): Volatility 추정 ===
    volatility = estimate_volatility_from_atr(atr, current_price)

    # === FIXED (C): Drift 추정 (최소주의) ===
    # FIXED (P0-5): use_mean_reversion 전달
    drift = estimate_drift_from_indicators(
        momentum=momentum,
        rsi=rsi,
        div_strength=div_strength,
        wyckoff_phase=wyckoff_phase,
        phase_confidence=phase_confidence,
        is_long=is_long,
        use_mean_reversion=use_mean_reversion
    )

    # === FIXED (A): 확률 계산 ===
    if use_monte_carlo:
        # Monte Carlo (시간 제한 반영)
        # FIXED (P0-7): seed 전달로 재현성 확보
        p_tp, p_sl, p_timeout = monte_carlo_barrier_probability(
            drift=drift,
            volatility=volatility,
            tp_distance=tp_log_dist,
            sl_distance=sl_log_dist,
            max_bars=max_hold_bars,
            n_simulations=n_simulations,
            seed=seed
        )
    else:
        # Analytic (무한 시간, 빠름)
        p_tp, p_sl = first_passage_probability_analytic(
            drift=drift,
            volatility=volatility,
            tp_distance=tp_log_dist,
            sl_distance=sl_log_dist
        )
        p_timeout = 0.0

    # === FIXED (E): EV 계산 ===
    ev_r, ev_pct = calculate_ev_with_costs(
        p_tp=p_tp,
        p_sl=p_sl,
        p_timeout=p_timeout,
        r_ratio=r_ratio,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        entry_fee_pct=entry_fee_pct,
        exit_fee_pct=exit_fee_pct,
        slippage_pct=slippage_pct
    )

    # === 신뢰도 계산 ===
    # 지표 일관성 기반
    signals = [
        np.sign(momentum) if abs(momentum) > 0.1 else 0,
        1 if rsi > 55 else -1 if rsi < 45 else 0,
        np.sign(div_strength) if abs(div_strength) > 0.1 else 0
    ]
    non_zero = [s for s in signals if s != 0]

    if len(non_zero) == 0:
        direction_confidence = 0.3
    elif all(s > 0 for s in non_zero) or all(s < 0 for s in non_zero):
        direction_confidence = 0.7
    else:
        direction_confidence = 0.4

    confidence = direction_confidence * phase_confidence

    return BarrierProbability(
        p_tp=p_tp,
        p_sl=p_sl,
        p_timeout=p_timeout,
        ev_r=ev_r,
        ev_pct=ev_pct,
        drift=drift,
        volatility=volatility,
        r_ratio=r_ratio,
        confidence=confidence
    )


def _default_probability() -> BarrierProbability:
    """기본값 (입력 오류 시)"""
    return BarrierProbability(
        p_tp=0.5,
        p_sl=0.5,
        p_timeout=0.0,
        ev_r=0.0,
        ev_pct=0.0,
        drift=0.0,
        volatility=0.01,
        r_ratio=1.0,
        confidence=0.0
    )


# ============================================================
# 편의 함수 (레거시 호환)
# ============================================================

def quick_probability(
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
    atr_pct: float,
    bullish_score: float,
    is_long: bool = True,
    max_hold_bars: int = 48
) -> float:
    """
    빠른 확률 계산 (간소화 버전)

    Args:
        entry_price: 진입 가격
        tp_pct: TP 비율 (예: 0.03 = 3%)
        sl_pct: SL 비율 (예: 0.015 = 1.5%)
        atr_pct: ATR 비율
        bullish_score: 상승 점수 (-1 ~ 1)
        is_long: 롱 여부
        max_hold_bars: 최대 보유 봉

    Returns:
        p_tp: TP 도달 확률
    """
    if is_long:
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

    result = calculate_barrier_probability(
        current_price=entry_price,
        tp_price=tp_price,
        sl_price=sl_price,
        atr=entry_price * atr_pct,
        max_hold_bars=max_hold_bars,
        momentum=bullish_score,
        rsi=50 + bullish_score * 30,
        is_long=is_long,
        use_monte_carlo=False  # 빠른 버전은 analytic
    )

    return result.p_tp


# 레거시 호환 함수들
def first_passage_probability(
    current_price: float,
    tp_price: float,
    sl_price: float,
    drift: float,
    volatility: float,
    is_long: bool = True
) -> Tuple[float, float]:
    """레거시 호환 (무한시간 analytic)"""
    if is_long:
        tp_dist = np.log(tp_price / current_price)
        sl_dist = np.log(current_price / sl_price)
    else:
        tp_dist = np.log(current_price / tp_price)
        sl_dist = np.log(sl_price / current_price)
        drift = -drift

    return first_passage_probability_analytic(drift, volatility, tp_dist, sl_dist)


def estimate_drift_from_momentum(
    momentum: float,
    energy: float,
    rsi: float,
    div_strength: float = 0.0,
    wyckoff_phase: str = "unknown",
    phase_confidence: float = 0.5
) -> float:
    """레거시 호환 (지표 기반 drift)"""
    return estimate_drift_from_indicators(
        momentum=momentum,
        rsi=rsi,
        div_strength=div_strength,
        wyckoff_phase=wyckoff_phase,
        phase_confidence=phase_confidence,
        is_long=True
    )
