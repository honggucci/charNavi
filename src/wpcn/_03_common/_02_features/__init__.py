"""
_02_features: 피처 엔지니어링 모듈
==================================

지표, 동적 파라미터, 타겟 생성 관련 모듈.

v3 추가:
- perf_optimized: Numba JIT 가속 + 캐싱 최적화
"""

# Dynamic Parameters
from .dynamic_params import DynamicParameterEngine, DynamicParams

# Performance Optimized (v3: Numba JIT)
from .perf_optimized import (
    FastDynamicEngine,
    VolatilityCache,
    fast_fft_cycle,
    fast_hilbert_phase,
    cached_volatility_regime,
    HAS_NUMBA,
)

# Indicators
from .indicators import (
    atr,
    rsi,
    stoch_rsi,
)

__all__ = [
    # Dynamic Params
    "DynamicParameterEngine",
    "DynamicParams",
    # Perf Optimized (v3)
    "FastDynamicEngine",
    "VolatilityCache",
    "fast_fft_cycle",
    "fast_hilbert_phase",
    "cached_volatility_regime",
    "HAS_NUMBA",
    # Indicators
    "atr",
    "rsi",
    "stoch_rsi",
]
