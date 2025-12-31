"""
전략별 파라미터 설정
Strategy-specific parameter configurations
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class StrategyConfig:
    """전략 설정"""
    name: str
    description: str

    # 타임프레임
    primary_timeframe: str = "15m"
    secondary_timeframe: str = "5m"

    # Wyckoff 파라미터
    box_lookback: int = 50
    volume_factor: float = 1.5

    # 피보나치 레벨
    fib_levels: tuple = (0.236, 0.382, 0.500, 0.618, 0.786)

    # 축적 모드
    use_accumulation: bool = True
    use_phase_c: bool = True

    # 롱/숏 설정
    long_enabled: bool = True
    short_enabled: bool = False  # 현물은 롱만

    # 시그널 필터
    use_rsi_filter: bool = True
    use_stoch_filter: bool = True
    use_divergence: bool = True

    # 비용
    fee_bps: float = 4.0  # 0.04%
    slippage_bps: float = 5.0  # 0.05%


# 현물 스윙 전략
SPOT_SWING_STRATEGY = StrategyConfig(
    name="spot_swing",
    description="현물 15분봉 스윙 전략 (롱 only)",
    primary_timeframe="15m",
    long_enabled=True,
    short_enabled=False,
    use_accumulation=True,
    use_phase_c=True,
)

# 선물 듀얼 타임프레임 전략
FUTURES_DUAL_TF_STRATEGY = StrategyConfig(
    name="futures_dual_tf",
    description="선물 15분봉 + 5분봉 듀얼 타임프레임 전략",
    primary_timeframe="15m",
    secondary_timeframe="5m",
    long_enabled=True,
    short_enabled=True,
    use_accumulation=True,
    use_phase_c=True,
    use_divergence=True,
)

# 선물 스캘핑 전략
FUTURES_SCALPING_STRATEGY = StrategyConfig(
    name="futures_scalping",
    description="선물 5분봉 스캘핑 전략",
    primary_timeframe="5m",
    long_enabled=True,
    short_enabled=True,
    use_accumulation=False,
    use_phase_c=False,
    use_divergence=True,
)


# 전략 설정 딕셔너리
STRATEGY_CONFIGS: Dict[str, StrategyConfig] = {
    "spot_swing": SPOT_SWING_STRATEGY,
    "futures_dual_tf": FUTURES_DUAL_TF_STRATEGY,
    "futures_scalping": FUTURES_SCALPING_STRATEGY,
}


def get_strategy_config(strategy_name: str) -> Optional[StrategyConfig]:
    """전략 설정 가져오기"""
    return STRATEGY_CONFIGS.get(strategy_name)
