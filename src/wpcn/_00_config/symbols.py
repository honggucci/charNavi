"""
코인별 파라미터 설정
Symbol-specific parameter configurations

각 코인의 특성에 맞는 최적화된 파라미터
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SymbolConfig:
    """코인별 설정"""
    symbol: str
    exchange: str = "binance"

    # 레버리지 설정
    spot_enabled: bool = True
    futures_enabled: bool = True
    max_leverage: float = 20.0
    recommended_leverage: float = 10.0

    # 포지션 사이징
    accumulation_pct: float = 0.03  # 축적 비율
    max_accumulation_pct: float = 0.15  # 최대 축적
    phase_c_pct: float = 0.20  # Phase C 진입 비율

    # TP/SL (레버리지 적용 전)
    tp_pct_15m: float = 0.015  # +1.5%
    sl_pct_15m: float = 0.010  # -1.0%
    tp_pct_5m: float = 0.008   # +0.8%
    sl_pct_5m: float = 0.005   # -0.5%

    # RSI/Stoch 필터
    rsi_oversold: float = 40.0
    rsi_overbought: float = 60.0
    stoch_oversold: float = 30.0
    stoch_overbought: float = 70.0

    # 시간 청산 (5분봉 기준)
    max_hold_bars_15m: int = 72  # 6시간
    max_hold_bars_5m: int = 36   # 3시간

    # 쿨다운
    accumulation_cooldown: int = 6  # 30분
    scalping_cooldown: int = 12     # 1시간


# BTC 설정 (높은 레버리지 허용)
BTC_CONFIG = SymbolConfig(
    symbol="BTC/USDT",
    max_leverage=20.0,
    recommended_leverage=15.0,

    # BTC는 변동성이 낮아 더 타이트한 TP/SL
    tp_pct_15m=0.015,
    sl_pct_15m=0.010,
    tp_pct_5m=0.008,
    sl_pct_5m=0.005,

    # RSI 필터 (약간 완화)
    rsi_oversold=45.0,
    rsi_overbought=55.0,
    stoch_oversold=40.0,
    stoch_overbought=60.0,
)

# ETH 설정 (중간 레버리지)
ETH_CONFIG = SymbolConfig(
    symbol="ETH/USDT",
    max_leverage=10.0,
    recommended_leverage=5.0,

    # ETH는 변동성이 높아 더 넓은 TP/SL
    tp_pct_15m=0.020,
    sl_pct_15m=0.012,
    tp_pct_5m=0.010,
    sl_pct_5m=0.006,

    # RSI 필터 (기본값)
    rsi_oversold=40.0,
    rsi_overbought=60.0,
    stoch_oversold=30.0,
    stoch_overbought=70.0,
)

# SOL 설정 (낮은 레버리지)
SOL_CONFIG = SymbolConfig(
    symbol="SOL/USDT",
    max_leverage=5.0,
    recommended_leverage=3.0,

    # SOL은 변동성이 매우 높음
    tp_pct_15m=0.025,
    sl_pct_15m=0.015,
    tp_pct_5m=0.012,
    sl_pct_5m=0.008,

    # RSI 필터 (더 엄격)
    rsi_oversold=35.0,
    rsi_overbought=65.0,
    stoch_oversold=25.0,
    stoch_overbought=75.0,
)


# 코인별 설정 딕셔너리
SYMBOL_CONFIGS: Dict[str, SymbolConfig] = {
    "BTC/USDT": BTC_CONFIG,
    "ETH/USDT": ETH_CONFIG,
    "SOL/USDT": SOL_CONFIG,
}


def get_symbol_config(symbol: str) -> SymbolConfig:
    """코인별 설정 가져오기"""
    if symbol in SYMBOL_CONFIGS:
        return SYMBOL_CONFIGS[symbol]

    # 기본 설정 반환
    return SymbolConfig(symbol=symbol)
