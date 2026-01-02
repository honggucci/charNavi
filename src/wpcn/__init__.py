"""
WPCN Backtester - Wyckoff Phase Chart Navigator

폴더 구조:
    _00_config/     - 설정 파일
    _01_crypto/     - 암호화폐 거래소별 모듈
    _02_data/       - 데이터 로딩/저장
    _03_common/     - 공통 모듈
    _04_execution/  - 백테스트 실행 엔진
    _05_probability/ - 확률 계산
    _06_engine/     - 엔진 레거시
    _07_reporting/  - 리포팅
    _08_tuning/     - 파라미터 튜닝
    _09_cli/        - CLI
    _10_ai_team/    - AI 팀
    _99_legacy/     - 레거시
"""
__all__ = ["__version__"]
__version__ = "0.1.0"

# =============================================================================
# Backward Compatibility Layer
# 구버전 import 경로 (wpcn.core, wpcn.data 등) 지원
# =============================================================================
import sys
from types import ModuleType

# Create compatibility modules
class _CompatModule(ModuleType):
    """Compatibility module that redirects to new paths"""
    def __init__(self, name, target_module):
        super().__init__(name)
        self._target = target_module

    def __getattr__(self, name):
        return getattr(self._target, name)

# Lazy import redirects
_redirects = {
    # Core types
    'wpcn.core': '_03_common._01_core',
    'wpcn.core.types': '_03_common._01_core.types',

    # Config
    'wpcn.config': '_00_config.config',

    # Common (alias)
    'wpcn.common': '_03_common',
    'wpcn.common.cost': '_03_common.cost',

    # Data
    'wpcn.data': '_02_data',
    'wpcn.data.ccxt_fetch': '_02_data.ccxt_fetch',
    'wpcn.data.loaders': '_02_data.loaders',
    'wpcn.data.resample': '_02_data.resample',

    # Engine
    'wpcn.engine': '_06_engine',
    'wpcn.engine.backtest': '_06_engine.backtest',
    'wpcn.engine.navigation': '_06_engine.navigation',
    'wpcn.engine.run_manager': '_06_engine.run_manager',

    # Execution
    'wpcn.execution': '_04_execution',
    'wpcn.execution.broker_sim': '_04_execution.broker_sim',
    'wpcn.execution.cost': '_04_execution.cost',
    'wpcn.execution.futures_backtest': '_99_legacy.futures_backtest',
    'wpcn.execution.futures_backtest_v3': '_99_legacy.futures_backtest_v3',
    'wpcn.execution.futures_backtest_v4': '_99_legacy.futures_backtest_v4',
    'wpcn.execution.futures_backtest_v6': '_99_legacy.futures_backtest_v6',
    'wpcn.execution.futures_backtest_v7': '_99_legacy.futures_backtest_v7',
    'wpcn.execution.futures_backtest_v8': '_99_legacy.futures_backtest_v8',

    # Features
    'wpcn.features': '_03_common._02_features',
    'wpcn.features.indicators': '_03_common._02_features.indicators',
    'wpcn.features.dynamic_params': '_03_common._02_features.dynamic_params',

    # Gate
    'wpcn.gate': '_03_common._09_gate',
    'wpcn.gate.regime': '_03_common._09_gate.regime',

    # Metrics
    'wpcn.metrics': '_03_common._05_metrics',
    'wpcn.metrics.performance': '_03_common._05_metrics.performance',

    # Probability
    'wpcn.probability': '_05_probability',
    'wpcn.probability.barrier': '_05_probability.barrier',
    'wpcn.probability.calibration': '_05_probability.calibration',

    # Reporting
    'wpcn.reporting': '_07_reporting',
    'wpcn.reporting.charts': '_07_reporting.charts',
    'wpcn.reporting.review_packet': '_07_reporting.review_packet',

    # Tuning
    'wpcn.tuning': '_08_tuning',
    'wpcn.tuning.theta_space': '_08_tuning.theta_space',
    'wpcn.tuning.walk_forward': '_08_tuning.walk_forward',
    'wpcn.tuning.weekly_optimizer': '_08_tuning.weekly_optimizer',

    # Wyckoff
    'wpcn.wyckoff': '_03_common._03_wyckoff',
    'wpcn.wyckoff.box': '_03_common._03_wyckoff.box',
    'wpcn.wyckoff.events': '_03_common._03_wyckoff.events',
    'wpcn.wyckoff.phases': '_03_common._03_wyckoff.phases',

    # Navigation
    'wpcn.navigation': '_03_common._04_navigation',
}

class _CompatFinder:
    """Module finder for backward compatibility"""
    def find_module(self, fullname, path=None):
        if fullname in _redirects:
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        target_path = _redirects.get(fullname)
        if target_path:
            # Import the target module
            import importlib
            target_fullname = f'wpcn.{target_path}'
            try:
                target = importlib.import_module(target_fullname)
                # Create compat module
                compat = _CompatModule(fullname, target)
                sys.modules[fullname] = compat
                return compat
            except ImportError:
                pass

        raise ImportError(f"No module named '{fullname}'")

# Register the compatibility finder
sys.meta_path.insert(0, _CompatFinder())
