from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from enum import Enum


class ExecutionProfile(Enum):
    """실행 프로파일 타입"""
    SPOT_NAVIGATION = "spot_navigation"      # 현물 와이코프 네비게이션
    SPOT_SCALPING = "spot_scalping"          # 현물 단타
    FUTURES_NAVIGATION = "futures_navigation" # 선물 와이코프 네비게이션
    FUTURES_SCALPING = "futures_scalping"    # 선물 단타


@dataclass(frozen=True)
class BacktestCosts:
    """
    거래 비용 설정

    현실적 비용 가이드:
    - Spot BTC (Maker):  fee=7.5bps, slip=3bps  → 총 10.5bps (0.105%)
    - Spot ALT (Taker):  fee=10bps,  slip=10bps → 총 20bps   (0.20%)
    - Futures (Maker):   fee=4bps,   slip=5bps  → 총 9bps    (0.09%)

    주의: 왕복 비용이 TP보다 크면 구조적 손실!
    """
    fee_bps: float
    slippage_bps: float

    def total_one_way_pct(self) -> float:
        """편도 총 비용 (%)"""
        return (self.fee_bps + self.slippage_bps) / 100

    def total_round_trip_pct(self) -> float:
        """왕복 총 비용 (%)"""
        return self.total_one_way_pct() * 2


# 프로파일별 권장 비용 설정
COSTS_SPOT_BTC = BacktestCosts(fee_bps=7.5, slippage_bps=3.0)      # 0.105% 편도
COSTS_SPOT_ALT = BacktestCosts(fee_bps=10.0, slippage_bps=10.0)    # 0.20% 편도
COSTS_FUTURES = BacktestCosts(fee_bps=4.0, slippage_bps=5.0)       # 0.09% 편도
COSTS_CONSERVATIVE = BacktestCosts(fee_bps=10.0, slippage_bps=15.0) # 0.25% 편도 (보수적)


@dataclass(frozen=True)
class BacktestConfig:
    """
    백테스트 설정

    Gate 철학: "안 들어가는 날이 많아져야 계좌가 산다"
    목표 No-trade 비율: 30% 이상
    """
    # portfolio + exits
    initial_equity: float = 1.0
    max_hold_bars: int = 192          # Navigation: 48시간 (15m*192), Scalping은 30으로 오버라이드
    tp1_frac: float = 0.5
    use_tp2: bool = True

    # WPCN gates (Unknown / No-Trade) - 상향 조정됨
    conf_min: float = 0.65            # 0.50 → 0.65 (쓰레기 신호 필터링)
    edge_min: float = 0.70            # 0.60 → 0.70 (엣지 없으면 진입 금지)
    confirm_bars: int = 2             # 1 → 2 (휩쏘 방지)
    reclaim_hold_bars: int = 2        # NEW: reclaim 후 박스 내부 유지 확인

    # regime safety
    chaos_ret_atr: float = 3.0        # CHAOS 레짐 판정: |ret| > 3*ATR → 진입 금지
    adx_len: int = 14
    adx_trend: float = 20.0


# 프로파일별 BacktestConfig 프리셋
CONFIG_NAVIGATION = BacktestConfig(
    max_hold_bars=192,    # 48시간
    conf_min=0.65,
    edge_min=0.70,
    confirm_bars=2,
    reclaim_hold_bars=2
)

CONFIG_SCALPING = BacktestConfig(
    max_hold_bars=30,     # 7.5시간
    conf_min=0.60,        # 단타는 약간 완화
    edge_min=0.65,
    confirm_bars=1,       # 단타는 빠른 진입
    reclaim_hold_bars=1
)


@dataclass(frozen=True)
class Theta:
    """
    와이코프 신호 감지 파라미터

    박스 + reclaim 계약:
    - box_L: 박스 길이 (15m*96 = 24시간)
    - N_reclaim: reclaim 허용 기간 (3봉 = 45분)
    - m_bw: 박스폭 비율 (reclaim 레벨 = box_edge + m_bw*box_width)
    """
    # box + reclaim contract
    pivot_lr: int         # 피벗 좌우 바 (스윙 감지)
    box_L: int            # 박스 길이 (15m*96 = 24시간)
    m_freeze: int         # 박스 프리즈 기간
    atr_len: int          # ATR 계산 기간
    x_atr: float          # 돌파 임계값 (ATR 배수)
    m_bw: float           # 박스폭 비율 (reclaim 레벨)
    N_reclaim: int        # reclaim 허용 기간 (봉 수)

    # execution realism knobs
    N_fill: int           # 주문 체결 허용 기간
    F_min: float          # Fill Probability Minimum (0.0~1.0)
                          # 계산: volume_ratio * price_proximity
                          # 용도: pending order가 F_min 미만이면 취소

@dataclass(frozen=True)
class RunConfig:
    exchange_id: str
    symbol: str
    timeframe: str
    days: int
    costs: BacktestCosts
    bt: BacktestConfig
    use_tuning: bool
    wf: Dict[str, Any]  # walk-forward settings
    mtf: List[str]      # e.g., ["15m","1h","4h","1d"]
    data_path: Optional[str] = None  # if provided, skip fetching

@dataclass(frozen=True)
class EventSignal:
    t_signal: Any
    side: str    # "long" | "short"
    event: str   # "spring" | "utad"
    entry_price: float
    stop_price: float
    tp1: float
    tp2: float
    meta: Dict[str, Any]
