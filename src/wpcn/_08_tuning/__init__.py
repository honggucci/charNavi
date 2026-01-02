"""
_08_tuning: 파라미터 튜닝 및 최적화 모듈 (v2.2)
===============================================

파라미터 단위 통일: 시간 관련 파라미터는 "분(minutes)" 단위로 저장,
런타임에 타임프레임에 맞게 "봉(bars)"으로 변환.

v2.2 변경사항:
- A/B Testing (Champion/Challenger) 버전 관리
- 자동 승격/롤백 로직 추가
- ParamVersionManager 모듈 추가

v2.1 변경사항:
- OOS (Out-of-Sample) 실시간 추적 통합
- ACTIVE 버전 관리 (set_active, get_active)
- 민감도 분석 (SensitivityAnalyzer)

모듈 구성:
----------
핵심 모듈:
- param_schema.py: 파라미터 스키마 정의 (TUNABLE/FIXED 구분, 단위 변환)
- param_store.py: 최적화된 파라미터 저장/로드 (v2 포맷, 신뢰도 지원)
- param_versioning.py: Champion/Challenger 버전 관리 (v2.2)
- param_optimizer.py: Grid/Random/Bayesian 옵티마이저
- walk_forward.py: Walk-Forward 최적화 (15m 네비게이션 공식)

분석 모듈:
- cycle_analyzer.py: Wyckoff 사이클 FFT 분석
- adaptive_space.py: 사이클 기반 적응형 탐색 공간
- sensitivity_analyzer.py: 파라미터 민감도 분석 (v2.1)
- oos_tracker.py: OOS 실시간 성능 추적 (v2.1)

실행 모듈:
- scheduler.py: 주간 자동 최적화 스케줄러 (v2.2: A/B Testing 통합)
- run_tuning.py: 통합 튜닝 파이프라인

분리된 모듈:
- scalping/: 5분봉 스캘핑 전용 (weekly_optimizer.py)
- legacy/: 폐기된 코드

Usage:
------
    # 15m 네비게이션 최적화 (공식)
    python -m wpcn._08_tuning.run_tuning --symbol BTC-USDT --timeframe 15m

    # 주간 자동 최적화 (OOS 확정 + A/B 비교 + 튜닝 + Challenger 등록)
    python -m wpcn._08_tuning.scheduler --run-now --market spot --timeframe 15m

    # 파라미터 변환 예시
    from wpcn._08_tuning import minutes_to_bars, convert_params_to_bars
    box_L = minutes_to_bars(1440, "15m")  # 1440분 → 96봉

    # OOS 추적 예시
    from wpcn._08_tuning import OOSTracker
    tracker = OOSTracker(market="spot")
    tracker.finalize_last_week_oos("BTC-USDT", "15m")

    # 버전 관리 예시
    from wpcn._08_tuning import get_version_manager
    manager = get_version_manager("BTC-USDT", "futures", "15m")
    action, comparison = manager.auto_decide()
"""

# ============================================================
# 파라미터 스키마 (v2 핵심)
# ============================================================
from .param_schema import (
    # 상수
    TUNABLE_PARAMS_MINUTES,
    TUNABLE_PARAMS_RATIO,
    FIXED_PARAMS,
    TF_MINUTES,
    # 변환 함수
    minutes_to_bars,
    bars_to_minutes,
    convert_params_to_bars,
    convert_params_to_minutes,
    # 검증
    validate_tunable_params,
    is_fixed_param,
    get_all_tunable_keys,
    # 기본값
    DefaultParams,
    DEFAULT_PARAMS,
)

# ============================================================
# 파라미터 저장소 (v2)
# ============================================================
from .param_store import (
    # 클래스
    ParamStore,
    OptimizedParams,
    ParamConfidence,
    ParamPerformance,
    SCHEMA_VERSION,
    # 함수
    get_param_store,
    save_optimized_params,
    load_optimized_params,
    # v2: load_reliable_as_bars는 ParamStore 메서드로 제공
)

# ============================================================
# 옵티마이저 (v3: Optuna TPE + Constraint Propagation)
# ============================================================
from .param_optimizer import (
    ParamOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    # v3 추가
    OptunaOptimizer,
    OptimizationResult,
    SamplingStats,
    ConstraintRegistry,
    ParamConstraint,
    get_optimizer,
    get_constraint_registry,
    # v3.1: Optuna availability check
    HAS_OPTUNA,
    get_optuna_status,
)

# ============================================================
# 차원 축소 (v3: Sensitivity 기반 자동화)
# ============================================================
from .param_reducer import (
    ParamReducer,
    ReductionConfig,
    ReducedParamSpace,
    auto_reduce_for_next_week,
)

# ============================================================
# Walk-Forward 최적화 (15m 네비게이션 공식)
# ============================================================
from .walk_forward import (
    WalkForwardConfig,
    WalkForwardOptimizer,
    WalkForwardResult,
    run_walk_forward_v8,
    V8_PARAM_SPACE,
)

# ============================================================
# 적응형 공간 생성
# ============================================================
from .adaptive_space import (
    AdaptiveParamSpace,
    ParamHints,
    generate_adaptive_space,
    load_analysis_and_generate_space,
    get_adaptive_param_space,
    build_theta_and_config,
    validate_params,
    create_constrained_sampler,
    THETA_KEYS,
    CONFIG_KEYS,
)

# ============================================================
# 사이클 분석
# ============================================================
from .cycle_analyzer import run_analysis as run_cycle_analysis

# ============================================================
# 민감도 분석 (v2.1)
# ============================================================
from .sensitivity_analyzer import (
    SensitivityAnalyzer,
    SensitivityResult,
    ParamSensitivity,
    compute_sensitivity_score,
    classify_params_by_stability,
)

# ============================================================
# OOS 실시간 추적 (v2.1)
# ============================================================
from .oos_tracker import (
    OOSTracker,
    OOSResult,
    compute_oos_adjusted_confidence,
)

# ============================================================
# 버전 관리 (v2.2 A/B Testing)
# ============================================================
from .param_versioning import (
    ParamVersionManager,
    VersionInfo,
    ComparisonResult,
    get_version_manager,
)

# ============================================================
# 스케줄러 (lazy import 권장)
# ============================================================
# from .scheduler import OptimizationScheduler  # 필요시 직접 import

__all__ = [
    # === 파라미터 스키마 (v2 핵심) ===
    "TUNABLE_PARAMS_MINUTES",
    "TUNABLE_PARAMS_RATIO",
    "FIXED_PARAMS",
    "TF_MINUTES",
    "minutes_to_bars",
    "bars_to_minutes",
    "convert_params_to_bars",
    "convert_params_to_minutes",
    "validate_tunable_params",
    "is_fixed_param",
    "get_all_tunable_keys",
    "DefaultParams",
    "DEFAULT_PARAMS",

    # === 파라미터 저장소 ===
    "ParamStore",
    "OptimizedParams",
    "ParamConfidence",
    "ParamPerformance",
    "SCHEMA_VERSION",
    "get_param_store",
    "save_optimized_params",
    "load_optimized_params",

    # === 옵티마이저 (v3: Optuna + Constraint) ===
    "ParamOptimizer",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "BayesianOptimizer",
    "OptunaOptimizer",
    "OptimizationResult",
    "SamplingStats",
    "ConstraintRegistry",
    "ParamConstraint",
    "get_optimizer",
    "get_constraint_registry",
    # v3.1: Optuna availability
    "HAS_OPTUNA",
    "get_optuna_status",

    # === 차원 축소 (v3) ===
    "ParamReducer",
    "ReductionConfig",
    "ReducedParamSpace",
    "auto_reduce_for_next_week",

    # === Walk-Forward ===
    "WalkForwardConfig",
    "WalkForwardOptimizer",
    "WalkForwardResult",
    "run_walk_forward_v8",
    "V8_PARAM_SPACE",

    # === 적응형 공간 ===
    "AdaptiveParamSpace",
    "ParamHints",
    "generate_adaptive_space",
    "load_analysis_and_generate_space",
    "get_adaptive_param_space",
    "build_theta_and_config",
    "validate_params",
    "create_constrained_sampler",
    "THETA_KEYS",
    "CONFIG_KEYS",

    # === 사이클 분석 ===
    "run_cycle_analysis",

    # === 민감도 분석 (v2.1) ===
    "SensitivityAnalyzer",
    "SensitivityResult",
    "ParamSensitivity",
    "compute_sensitivity_score",
    "classify_params_by_stability",

    # === OOS 실시간 추적 (v2.1) ===
    "OOSTracker",
    "OOSResult",
    "compute_oos_adjusted_confidence",

    # === 버전 관리 (v2.2 A/B Testing) ===
    "ParamVersionManager",
    "VersionInfo",
    "ComparisonResult",
    "get_version_manager",
]
