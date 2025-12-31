"""
차트 네비게이션 모듈
Chart Navigation Module

확률론적 와이코프 스나이퍼 매매를 위한 네비게이션 시스템
"""
from .fibonacci_confluence import (
    MTFFibonacciConfluence,
    ConfluenceZone,
    FibonacciLevel,
    SwingPoint,
    FIBONACCI_LEVELS,
    FIBONACCI_EXTENSIONS,
)

__all__ = [
    'MTFFibonacciConfluence',
    'ConfluenceZone',
    'FibonacciLevel',
    'SwingPoint',
    'FIBONACCI_LEVELS',
    'FIBONACCI_EXTENSIONS',
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'PhysicsValidator':
        from .physics_validator import PhysicsValidator
        return PhysicsValidator
    elif name == 'ProbabilityCalculator':
        from .probability_calculator import ProbabilityCalculator
        return ProbabilityCalculator
    elif name == 'ChartNavigator':
        from .chart_navigator import ChartNavigator
        return ChartNavigator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
