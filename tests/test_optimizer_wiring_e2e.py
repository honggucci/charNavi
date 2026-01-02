"""
E2E Optimizer Wiring Tests
==========================

Verifies that CLI --optimizer option actually routes to correct optimizer class.
Prevents regression where CLI shows "optuna" but internally uses "random".

Test Cases:
1. run_tuning CLI --optimizer parsing
2. WalkForwardConfig.optimizer_type propagation
3. walk_forward.py get_optimizer() call with correct type
4. get_optimizer() factory returns correct class
"""
import pytest
from unittest.mock import patch, MagicMock
import argparse


class TestGetOptimizerFactory:
    """Test get_optimizer() factory returns correct classes"""

    def test_random_returns_random_optimizer(self):
        from wpcn._08_tuning.param_optimizer import get_optimizer, RandomSearchOptimizer
        opt = get_optimizer("random")
        assert isinstance(opt, RandomSearchOptimizer)

    def test_optuna_returns_correct_optimizer(self):
        """Optuna returns OptunaOptimizer if installed, else RandomSearchOptimizer"""
        from wpcn._08_tuning.param_optimizer import (
            get_optimizer, OptunaOptimizer, RandomSearchOptimizer, HAS_OPTUNA
        )
        opt = get_optimizer("optuna")
        if HAS_OPTUNA:
            assert isinstance(opt, OptunaOptimizer)
        else:
            assert isinstance(opt, RandomSearchOptimizer)
            assert "optuna fallback" in opt.actual_type

    def test_grid_returns_grid_optimizer(self):
        from wpcn._08_tuning.param_optimizer import get_optimizer, GridSearchOptimizer
        opt = get_optimizer("grid")
        assert isinstance(opt, GridSearchOptimizer)

    def test_bayesian_returns_bayesian_optimizer(self):
        from wpcn._08_tuning.param_optimizer import get_optimizer, BayesianOptimizer
        opt = get_optimizer("bayesian")
        assert isinstance(opt, BayesianOptimizer)

    def test_invalid_type_falls_back_to_random(self):
        """Invalid optimizer type should fallback to RandomSearch (graceful degradation)"""
        from wpcn._08_tuning.param_optimizer import get_optimizer, RandomSearchOptimizer
        opt = get_optimizer("invalid_optimizer_type")
        assert isinstance(opt, RandomSearchOptimizer), \
            "Invalid type should fallback to RandomSearchOptimizer"


class TestRunTuningCLI:
    """Test run_tuning.py CLI parsing"""

    def test_optimizer_arg_exists(self):
        """--optimizer argument should exist in CLI"""
        from wpcn._08_tuning import run_tuning
        import inspect
        source = inspect.getsource(run_tuning)
        assert "--optimizer" in source, "CLI --optimizer option missing"

    def test_optimizer_choices(self):
        """--optimizer should accept random, optuna, bayesian, grid"""
        from wpcn._08_tuning import run_tuning
        import inspect
        source = inspect.getsource(run_tuning)
        # Check all 4 choices are defined
        assert "random" in source
        assert "optuna" in source
        assert "bayesian" in source
        assert "grid" in source


class TestWalkForwardConfig:
    """Test WalkForwardConfig passes optimizer_type correctly"""

    def test_config_has_optimizer_type(self):
        from wpcn._08_tuning.walk_forward import WalkForwardConfig
        cfg = WalkForwardConfig(optimizer_type="optuna")
        assert cfg.optimizer_type == "optuna"

    def test_config_default_is_random(self):
        from wpcn._08_tuning.walk_forward import WalkForwardConfig
        cfg = WalkForwardConfig()
        assert cfg.optimizer_type == "random"


class TestWalkForwardOptimizerSelection:
    """Test walk_forward.py uses get_optimizer with config type"""

    def test_walk_forward_imports_get_optimizer(self):
        """walk_forward.py should import get_optimizer"""
        from wpcn._08_tuning import walk_forward
        import inspect
        source = inspect.getsource(walk_forward)
        assert "get_optimizer" in source, "walk_forward missing get_optimizer import"

    def test_walk_forward_calls_get_optimizer(self):
        """walk_forward should call optimizer = get_optimizer(...)"""
        from wpcn._08_tuning import walk_forward
        import inspect
        source = inspect.getsource(walk_forward)
        # Check pattern: optimizer = get_optimizer(
        assert "optimizer = get_optimizer" in source, \
            "walk_forward.py not calling get_optimizer factory"


class TestE2EOptimizerWiring:
    """E2E test: CLI option -> actual optimizer class"""

    def test_optuna_option_with_fallback_tracking(self):
        """
        Critical E2E test:
        --optimizer optuna must result in OptunaOptimizer if installed,
        or RandomSearchOptimizer with actual_type='random (optuna fallback)' if not.
        """
        from wpcn._08_tuning.walk_forward import WalkForwardConfig
        from wpcn._08_tuning.param_optimizer import (
            get_optimizer, OptunaOptimizer, RandomSearchOptimizer, HAS_OPTUNA
        )

        # Simulate CLI option
        cfg = WalkForwardConfig(optimizer_type="optuna")

        # Get optimizer as walk_forward would
        optimizer = get_optimizer(cfg.optimizer_type)

        if HAS_OPTUNA:
            assert isinstance(optimizer, OptunaOptimizer), \
                f"Expected OptunaOptimizer, got {type(optimizer).__name__}"
            assert optimizer.actual_type == "optuna"
        else:
            # Fallback case: optuna not installed
            assert isinstance(optimizer, RandomSearchOptimizer), \
                f"Expected RandomSearchOptimizer (fallback), got {type(optimizer).__name__}"
            assert "optuna fallback" in optimizer.actual_type, \
                f"actual_type should indicate fallback, got: {optimizer.actual_type}"

    def test_grid_option_uses_grid_optimizer(self):
        """--optimizer grid must result in GridSearchOptimizer"""
        from wpcn._08_tuning.walk_forward import WalkForwardConfig
        from wpcn._08_tuning.param_optimizer import get_optimizer, GridSearchOptimizer

        cfg = WalkForwardConfig(optimizer_type="grid")
        optimizer = get_optimizer(cfg.optimizer_type)

        assert isinstance(optimizer, GridSearchOptimizer), \
            f"Expected GridSearchOptimizer, got {type(optimizer).__name__}"

    def test_random_option_uses_random_optimizer(self):
        """--optimizer random must result in RandomSearchOptimizer"""
        from wpcn._08_tuning.walk_forward import WalkForwardConfig
        from wpcn._08_tuning.param_optimizer import get_optimizer, RandomSearchOptimizer

        cfg = WalkForwardConfig(optimizer_type="random")
        optimizer = get_optimizer(cfg.optimizer_type)

        assert isinstance(optimizer, RandomSearchOptimizer), \
            f"Expected RandomSearchOptimizer, got {type(optimizer).__name__}"


class TestConstraintRegistryExists:
    """Verify ConstraintRegistry is properly exported"""

    def test_constraint_registry_importable(self):
        from wpcn._08_tuning import ConstraintRegistry
        assert ConstraintRegistry is not None

    def test_get_constraint_registry_importable(self):
        from wpcn._08_tuning import get_constraint_registry
        assert callable(get_constraint_registry)


class TestParamReducerExists:
    """Verify ParamReducer is properly exported"""

    def test_param_reducer_importable(self):
        from wpcn._08_tuning import ParamReducer
        assert ParamReducer is not None

    def test_reduction_config_importable(self):
        from wpcn._08_tuning import ReductionConfig
        assert ReductionConfig is not None


class TestOptunaStatusTracking:
    """v3.1: Verify Optuna availability tracking"""

    def test_has_optuna_exported(self):
        from wpcn._08_tuning import HAS_OPTUNA
        assert isinstance(HAS_OPTUNA, bool)

    def test_get_optuna_status_exported(self):
        from wpcn._08_tuning import get_optuna_status
        status = get_optuna_status()
        assert "optuna_available" in status
        assert "optuna_version" in status

    def test_optimizer_has_actual_type(self):
        """All optimizers from factory should have actual_type attribute"""
        from wpcn._08_tuning.param_optimizer import get_optimizer
        for opt_type in ["random", "grid", "bayesian", "optuna"]:
            opt = get_optimizer(opt_type)
            assert hasattr(opt, "actual_type"), \
                f"Optimizer {opt_type} missing actual_type attribute"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
