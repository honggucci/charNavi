"""
파라미터 최적화 모듈 (v3 - Institutional Grade)
==============================================

P1-A: Constraint Propagation 일반화 (하드코딩 제거)
P1-B: Optuna 기반 TPE Optimizer 추가

Features:
- Grid Search: 모든 조합 탐색
- Random Search: Rejection Sampling + Constraint Propagation
- Bayesian (skopt): Gaussian Process 기반
- TPE (Optuna): Tree-structured Parzen Estimator (권장!)

v3 개선사항:
1. ConstraintRegistry: 제약조건 중앙 관리
2. FeasibleSampler: 제약 전파로 "될 놈만" 샘플링
3. OptunaOptimizer: TPE 기반 고효율 탐색
4. Acceptance Rate 로깅: 병목 모니터링
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Tuple, Optional
import itertools
import numpy as np
import pandas as pd
from datetime import datetime


# ============================================================
# Optuna Availability Check (v3.1: 명시적 로깅)
# ============================================================
try:
    import optuna
    HAS_OPTUNA = True
    OPTUNA_VERSION = optuna.__version__
except ImportError:
    HAS_OPTUNA = False
    OPTUNA_VERSION = None


def get_optuna_status() -> dict:
    """Optuna 설치 상태 반환 (결과 JSON 기록용)"""
    return {
        "optuna_available": HAS_OPTUNA,
        "optuna_version": OPTUNA_VERSION,
    }


# ============================================================
# Data Classes
# ============================================================

@dataclass
class OptimizationResult:
    """최적화 결과"""
    best_params: Dict
    best_score: float
    all_results: List[Dict]
    n_iterations: int
    # v3: 메타데이터 추가
    optimizer_type: str = "unknown"
    acceptance_rate: float = 1.0  # rejection sampling 성공률
    total_samples: int = 0        # 총 샘플링 시도 횟수
    elapsed_seconds: float = 0.0


@dataclass
class SamplingStats:
    """샘플링 통계 (병목 모니터링)"""
    total_attempts: int = 0
    valid_samples: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        if self.total_attempts == 0:
            return 1.0
        return self.valid_samples / self.total_attempts

    def log_rejection(self, reason: str):
        self.total_attempts += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

    def log_accept(self):
        self.total_attempts += 1
        self.valid_samples += 1

    def report(self) -> str:
        lines = [
            f"Sampling Stats:",
            f"  Total Attempts: {self.total_attempts}",
            f"  Valid Samples: {self.valid_samples}",
            f"  Acceptance Rate: {self.acceptance_rate:.2%}",
        ]
        if self.rejection_reasons:
            lines.append("  Top Rejection Reasons:")
            sorted_reasons = sorted(self.rejection_reasons.items(), key=lambda x: -x[1])[:5]
            for reason, count in sorted_reasons:
                lines.append(f"    - {reason}: {count}")
        return "\n".join(lines)


# ============================================================
# Constraint Registry (P1-A: 하드코딩 제거)
# ============================================================

@dataclass
class ParamConstraint:
    """파라미터 제약조건 정의"""
    name: str
    check_fn: Callable[[Dict], Tuple[bool, str]]
    # 종속성 (이 제약이 어떤 파라미터에 의존하는지)
    depends_on: List[str] = field(default_factory=list)
    # feasible 범위 계산 함수 (constraint propagation용)
    propagate_fn: Optional[Callable[[Dict, Tuple], Tuple]] = None


class ConstraintRegistry:
    """
    제약조건 중앙 관리

    사용법:
        registry = ConstraintRegistry()
        registry.register("m_freeze_box", lambda p: (p["m_freeze"] < p["box_L"], ...))
        registry.validate(params)  # 전체 검증
        registry.propagate("m_freeze", params, spec)  # 범위 조정
    """

    # 기본 제약조건 정의
    DEFAULT_CONSTRAINTS = [
        ParamConstraint(
            name="m_freeze_lt_box",
            check_fn=lambda p: (
                p.get("m_freeze", 32) < p.get("box_L", 96),
                f"m_freeze({p.get('m_freeze')}) >= box_L({p.get('box_L')})"
            ),
            depends_on=["box_L"],
            propagate_fn=lambda p, spec: (
                max(spec[0], p.get("box_L", 96) // 6),
                min(spec[1], int(p.get("box_L", 96) * 0.5))
            )
        ),
        ParamConstraint(
            name="m_freeze_ratio",
            check_fn=lambda p: (
                p.get("m_freeze", 32) <= p.get("box_L", 96) * 0.6,
                f"m_freeze({p.get('m_freeze')}) > box_L*0.6({p.get('box_L', 96)*0.6:.0f})"
            ),
            depends_on=["box_L"]
        ),
        ParamConstraint(
            name="pivot_lr_ratio",
            check_fn=lambda p: (
                p.get("pivot_lr", 4) <= p.get("box_L", 96) // 10,
                f"pivot_lr({p.get('pivot_lr')}) > box_L/10({p.get('box_L', 96)//10})"
            ),
            depends_on=["box_L"],
            propagate_fn=lambda p, spec: (
                spec[0],
                min(spec[1], max(spec[0], p.get("box_L", 96) // 10))
            )
        ),
        ParamConstraint(
            name="tp_gt_sl",
            check_fn=lambda p: (
                p.get("tp_pct", 0.015) >= p.get("sl_pct", 0.01),
                f"tp_pct({p.get('tp_pct')}) < sl_pct({p.get('sl_pct')})"
            ),
            depends_on=["sl_pct"],
            propagate_fn=lambda p, spec: (
                max(spec[0], p.get("sl_pct", 0.01) * 1.05),
                spec[1]
            )
        ),
        ParamConstraint(
            name="box_L_min_for_pivot",
            check_fn=lambda p: (
                p.get("box_L", 96) >= 20,
                f"box_L({p.get('box_L')}) < 20 (pivot_lr >= 2 불가)"
            ),
            depends_on=[],
            propagate_fn=lambda p, spec: (
                max(spec[0], 20),
                spec[1]
            )
        ),
    ]

    def __init__(self):
        self.constraints: Dict[str, ParamConstraint] = {}
        # 기본 제약조건 등록
        for c in self.DEFAULT_CONSTRAINTS:
            self.register(c)

    def register(self, constraint: ParamConstraint):
        """제약조건 등록"""
        self.constraints[constraint.name] = constraint

    def validate(self, params: Dict) -> Tuple[bool, str]:
        """전체 제약조건 검증"""
        for name, c in self.constraints.items():
            is_valid, reason = c.check_fn(params)
            if not is_valid:
                return False, reason
        return True, "ok"

    def propagate_bounds(
        self,
        param_name: str,
        current_params: Dict,
        original_spec: Tuple
    ) -> Tuple:
        """
        Constraint Propagation: 제약을 만족하도록 범위 조정

        Args:
            param_name: 조정할 파라미터 이름
            current_params: 이미 샘플링된 파라미터들
            original_spec: 원래 (min, max, type) 스펙

        Returns:
            조정된 (min, max) 범위
        """
        # 해당 파라미터에 영향 주는 제약조건 찾기
        propagation_map = {
            "box_L": ["box_L_min_for_pivot"],
            "m_freeze": ["m_freeze_lt_box", "m_freeze_ratio"],
            "pivot_lr": ["pivot_lr_ratio"],
            "tp_pct": ["tp_gt_sl"],
        }

        constraint_names = propagation_map.get(param_name, [])
        if not constraint_names:
            return original_spec[:2]

        new_min, new_max = original_spec[0], original_spec[1]

        for cname in constraint_names:
            c = self.constraints.get(cname)
            if c and c.propagate_fn:
                prop_min, prop_max = c.propagate_fn(current_params, (new_min, new_max))
                new_min = max(new_min, prop_min)
                new_max = min(new_max, prop_max)

        # 범위가 역전되면 최소한의 유효 범위 반환
        if new_max < new_min:
            new_max = new_min

        return new_min, new_max


# 글로벌 레지스트리
_constraint_registry = ConstraintRegistry()


def get_constraint_registry() -> ConstraintRegistry:
    return _constraint_registry


def validate_params(params: Dict) -> Tuple[bool, str]:
    """파라미터 검증 (외부 인터페이스)"""
    return _constraint_registry.validate(params)


# ============================================================
# Abstract Base Class
# ============================================================

class ParamOptimizer(ABC):
    """파라미터 최적화 기본 클래스"""

    @abstractmethod
    def optimize(
        self,
        param_space: Dict,
        objective_func: Callable[[Dict], float],
        n_iterations: int = 100
    ) -> OptimizationResult:
        """최적화 실행"""
        pass


# ============================================================
# Grid Search Optimizer
# ============================================================

class GridSearchOptimizer(ParamOptimizer):
    """
    Grid Search 최적화
    - 모든 조합 탐색
    - 단순하지만 조합 수가 많으면 느림
    """

    def optimize(
        self,
        param_space: Dict[str, List],
        objective_func: Callable[[Dict], float],
        n_iterations: int = None  # Grid Search는 무시
    ) -> OptimizationResult:
        """Grid Search 실행"""
        import time
        start_time = time.time()

        keys = list(param_space.keys())
        values = list(param_space.values())
        combinations = list(itertools.product(*values))

        print(f"Grid Search: {len(combinations)} combinations")

        all_results = []
        best_score = float('-inf')
        best_params = None
        valid_count = 0

        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))

            # 제약조건 검증
            is_valid, _ = validate_params(params)
            if not is_valid:
                continue

            valid_count += 1

            try:
                score = objective_func(params)
            except Exception as e:
                print(f"  Error with params {params}: {e}")
                score = float('-inf')

            all_results.append({
                "params": params,
                "score": score
            })

            if score > best_score:
                best_score = score
                best_params = params

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(combinations)}, Valid: {valid_count}, Best: {best_score:.4f}")

        elapsed = time.time() - start_time

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            n_iterations=len(combinations),
            optimizer_type="grid",
            acceptance_rate=valid_count / len(combinations) if combinations else 0,
            total_samples=len(combinations),
            elapsed_seconds=elapsed
        )


# ============================================================
# Random Search Optimizer (P1-A 개선)
# ============================================================

class RandomSearchOptimizer(ParamOptimizer):
    """
    Random Search 최적화 (v3)

    P1-A 개선사항:
    - Constraint Propagation으로 "될 놈만" 샘플링
    - Rejection Sampling은 폴백으로만 사용
    - Acceptance Rate 로깅
    """

    MAX_SAMPLE_ATTEMPTS = 200

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.stats = SamplingStats()
        self.registry = get_constraint_registry()

    def optimize(
        self,
        param_space: Dict[str, Tuple],
        objective_func: Callable[[Dict], float],
        n_iterations: int = 100
    ) -> OptimizationResult:
        """Random Search 실행"""
        import time
        start_time = time.time()

        print(f"Random Search (v3): {n_iterations} iterations")
        print(f"  Constraint Propagation enabled")

        self.stats = SamplingStats()  # 통계 리셋

        all_results = []
        best_score = float('-inf')
        best_params = None

        for i in range(n_iterations):
            try:
                params = self._sample_with_propagation(param_space)
            except ValueError as e:
                print(f"  Sampling failed at iteration {i}: {e}")
                continue

            try:
                score = objective_func(params)
            except Exception as e:
                print(f"  Error with params {params}: {e}")
                score = float('-inf')

            all_results.append({
                "params": params,
                "score": score
            })

            if score > best_score:
                best_score = score
                best_params = params

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{n_iterations}, Best: {best_score:.4f}, Accept: {self.stats.acceptance_rate:.1%}")

        elapsed = time.time() - start_time

        # 샘플링 통계 출력
        print(f"\n{self.stats.report()}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            n_iterations=n_iterations,
            optimizer_type="random_v3",
            acceptance_rate=self.stats.acceptance_rate,
            total_samples=self.stats.total_attempts,
            elapsed_seconds=elapsed
        )

    def _sample_with_propagation(self, param_space: Dict) -> Dict:
        """
        Constraint Propagation 기반 샘플링

        종속 순서: box_L → m_freeze → pivot_lr → sl_pct → tp_pct
        각 단계에서 범위를 제약 기반으로 조정
        """
        params = {}

        # 샘플링 순서 정의 (종속성 기반)
        sampling_order = [
            "box_L",      # 1순위: 독립 변수
            "m_freeze",   # 2순위: box_L에 종속
            "pivot_lr",   # 3순위: box_L에 종속
            "atr_len",    # 독립
            "x_atr",      # 독립
            "m_bw",       # 독립
            "sl_pct",     # 4순위: 독립이지만 tp_pct의 선행
            "tp_pct",     # 5순위: sl_pct에 종속
            "min_score",  # 독립
            "rsi_oversold",
            "rsi_overbought",
        ]

        # 정의된 순서대로 샘플링
        for param_name in sampling_order:
            if param_name not in param_space:
                continue

            spec = param_space[param_name]

            # Constraint Propagation: 범위 조정
            adj_min, adj_max = self.registry.propagate_bounds(param_name, params, spec)

            # 샘플링
            if len(spec) >= 3 and spec[-1] == 'int':
                adj_min = int(adj_min)
                adj_max = int(adj_max)
                if adj_max < adj_min:
                    adj_max = adj_min
                params[param_name] = int(self.rng.integers(adj_min, adj_max + 1))
            elif len(spec) >= 3 and spec[-1] == 'choice':
                options = spec[:-1]
                params[param_name] = self.rng.choice(options)
            else:
                if adj_max < adj_min:
                    adj_max = adj_min
                params[param_name] = float(self.rng.uniform(adj_min, adj_max))

        # 나머지 파라미터 (순서에 없는 것들)
        for name, spec in param_space.items():
            if name in params:
                continue

            if len(spec) >= 3 and spec[-1] == 'choice':
                options = spec[:-1]
                params[name] = self.rng.choice(options)
            elif len(spec) >= 3 and spec[-1] == 'int':
                params[name] = int(self.rng.integers(spec[0], spec[1] + 1))
            else:
                params[name] = float(self.rng.uniform(spec[0], spec[1]))

        # 최종 검증 (폴백)
        is_valid, reason = self.registry.validate(params)
        if is_valid:
            self.stats.log_accept()
            return params

        # Constraint propagation 실패 시 rejection sampling 폴백
        self.stats.log_rejection(reason)
        return self._rejection_sampling_fallback(param_space)

    def _rejection_sampling_fallback(self, param_space: Dict) -> Dict:
        """Rejection Sampling 폴백 (constraint propagation 실패 시)"""
        for attempt in range(self.MAX_SAMPLE_ATTEMPTS):
            params = self._sample_raw(param_space)
            is_valid, reason = self.registry.validate(params)
            if is_valid:
                self.stats.log_accept()
                return params
            self.stats.log_rejection(reason)

        raise ValueError(f"Cannot sample valid params after {self.MAX_SAMPLE_ATTEMPTS} rejection attempts")

    def _sample_raw(self, param_space: Dict) -> Dict:
        """순수 랜덤 샘플링 (제약 무시)"""
        params = {}
        for name, spec in param_space.items():
            if len(spec) >= 3 and spec[-1] == 'choice':
                params[name] = self.rng.choice(spec[:-1])
            elif len(spec) >= 3 and spec[-1] == 'int':
                params[name] = int(self.rng.integers(spec[0], spec[1] + 1))
            else:
                params[name] = float(self.rng.uniform(spec[0], spec[1]))
        return params


# ============================================================
# Bayesian Optimizer (skopt)
# ============================================================

class BayesianOptimizer(ParamOptimizer):
    """
    Bayesian Optimization (scikit-optimize)
    - Gaussian Process 기반
    - 제약조건: invalid는 패널티로 처리
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def optimize(
        self,
        param_space: Dict[str, Tuple],
        objective_func: Callable[[Dict], float],
        n_iterations: int = 50
    ) -> OptimizationResult:
        """Bayesian Optimization 실행"""
        import time
        start_time = time.time()

        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            print("scikit-optimize not installed. Falling back to Random Search.")
            return RandomSearchOptimizer().optimize(param_space, objective_func, n_iterations)

        # 파라미터 공간 변환
        dimensions = []
        param_names = []

        for name, spec in param_space.items():
            param_names.append(name)
            if len(spec) == 3 and spec[2] == 'int':
                dimensions.append(Integer(spec[0], spec[1], name=name))
            elif len(spec) > 2 and spec[-1] == 'choice':
                dimensions.append(Categorical(spec[:-1], name=name))
            else:
                dimensions.append(Real(spec[0], spec[1], name=name))

        all_results = []
        invalid_count = 0

        def objective(x):
            nonlocal invalid_count
            params = dict(zip(param_names, x))

            # 제약조건 검증
            is_valid, _ = validate_params(params)
            if not is_valid:
                invalid_count += 1
                return 1e6  # 큰 패널티

            try:
                score = objective_func(params)
            except Exception as e:
                print(f"  Error: {e}")
                score = float('-inf')

            all_results.append({"params": params, "score": score})
            return -score  # 최소화

        print(f"Bayesian Optimization (skopt): {n_iterations} iterations")

        result = gp_minimize(
            objective,
            dimensions,
            n_calls=n_iterations,
            random_state=self.random_state,
            verbose=True
        )

        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun

        elapsed = time.time() - start_time
        valid_count = n_iterations - invalid_count

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            n_iterations=n_iterations,
            optimizer_type="bayesian_skopt",
            acceptance_rate=valid_count / n_iterations if n_iterations > 0 else 0,
            total_samples=n_iterations,
            elapsed_seconds=elapsed
        )


# ============================================================
# Optuna TPE Optimizer (P1-B: 권장!)
# ============================================================

class OptunaOptimizer(ParamOptimizer):
    """
    Optuna TPE Optimizer (권장!)

    Tree-structured Parzen Estimator:
    - skopt보다 빠르고 효율적
    - 제약조건: Trial pruning으로 처리
    - 병렬화 지원 (옵션)
    """

    def __init__(self, seed: int = 42, n_startup_trials: int = 10):
        self.seed = seed
        self.n_startup_trials = n_startup_trials

    def optimize(
        self,
        param_space: Dict[str, Tuple],
        objective_func: Callable[[Dict], float],
        n_iterations: int = 50
    ) -> OptimizationResult:
        """Optuna TPE 최적화 실행"""
        import time
        start_time = time.time()

        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            print("Optuna not installed. Falling back to Random Search.")
            print("Install: pip install optuna")
            return RandomSearchOptimizer().optimize(param_space, objective_func, n_iterations)

        # Optuna 로깅 억제
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        all_results = []
        invalid_count = 0

        def optuna_objective(trial: optuna.Trial) -> float:
            nonlocal invalid_count
            params = {}

            # 종속 순서대로 샘플링 (constraint propagation)
            # box_L 먼저
            if "box_L" in param_space:
                spec = param_space["box_L"]
                low, high = max(spec[0], 20), spec[1]  # 최소 20
                if spec[-1] == 'int':
                    params["box_L"] = trial.suggest_int("box_L", low, high)
                else:
                    params["box_L"] = trial.suggest_float("box_L", low, high)

            box_L = params.get("box_L", 96)

            # m_freeze (box_L에 종속)
            if "m_freeze" in param_space:
                spec = param_space["m_freeze"]
                low = max(spec[0], box_L // 6)
                high = min(spec[1], int(box_L * 0.5))
                if high < low:
                    high = low
                params["m_freeze"] = trial.suggest_int("m_freeze", low, high)

            # pivot_lr (box_L에 종속)
            if "pivot_lr" in param_space:
                spec = param_space["pivot_lr"]
                low = spec[0]
                high = min(spec[1], max(low, box_L // 10))
                params["pivot_lr"] = trial.suggest_int("pivot_lr", low, high)

            # sl_pct 먼저
            if "sl_pct" in param_space:
                spec = param_space["sl_pct"]
                params["sl_pct"] = trial.suggest_float("sl_pct", spec[0], spec[1])

            sl_pct = params.get("sl_pct", 0.01)

            # tp_pct (sl_pct에 종속)
            if "tp_pct" in param_space:
                spec = param_space["tp_pct"]
                low = max(spec[0], sl_pct * 1.05)
                high = spec[1]
                if high < low:
                    high = low
                params["tp_pct"] = trial.suggest_float("tp_pct", low, high)

            # 나머지 독립 파라미터
            for name, spec in param_space.items():
                if name in params:
                    continue

                if len(spec) >= 3 and spec[-1] == 'choice':
                    params[name] = trial.suggest_categorical(name, spec[:-1])
                elif len(spec) >= 3 and spec[-1] == 'int':
                    params[name] = trial.suggest_int(name, spec[0], spec[1])
                else:
                    params[name] = trial.suggest_float(name, spec[0], spec[1])

            # 최종 제약조건 검증
            is_valid, reason = validate_params(params)
            if not is_valid:
                invalid_count += 1
                raise optuna.TrialPruned(reason)

            try:
                score = objective_func(params)
            except Exception as e:
                print(f"  Error: {e}")
                raise optuna.TrialPruned(str(e))

            all_results.append({"params": params, "score": score})
            return score  # Optuna는 maximize 모드 사용

        # Study 생성
        sampler = TPESampler(seed=self.seed, n_startup_trials=self.n_startup_trials)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"wpcn_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        print(f"Optuna TPE: {n_iterations} iterations")
        print(f"  Startup trials: {self.n_startup_trials}")

        study.optimize(
            optuna_objective,
            n_trials=n_iterations,
            show_progress_bar=True,
            catch=(Exception,)
        )

        elapsed = time.time() - start_time

        # 결과 추출
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value

        valid_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        print(f"\nOptuna Summary:")
        print(f"  Best Score: {best_score:.4f}")
        print(f"  Valid Trials: {valid_count}/{n_iterations}")
        print(f"  Pruned Trials: {invalid_count}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            n_iterations=n_iterations,
            optimizer_type="optuna_tpe",
            acceptance_rate=valid_count / n_iterations if n_iterations > 0 else 0,
            total_samples=n_iterations,
            elapsed_seconds=elapsed
        )


# ============================================================
# Factory Function
# ============================================================

def get_optimizer(
    optimizer_type: str = "random",
    seed: int = 42
) -> ParamOptimizer:
    """
    옵티마이저 팩토리

    Args:
        optimizer_type: "random", "grid", "bayesian", "optuna" (권장)
        seed: 랜덤 시드

    Returns:
        ParamOptimizer 인스턴스

    Note:
        optuna 요청 시 optuna 미설치면 RandomSearch로 fallback됨.
        실제 사용된 옵티마이저는 optimizer.actual_type으로 확인 가능.
    """
    if optimizer_type == "grid":
        opt = GridSearchOptimizer()
        opt.actual_type = "grid"
        return opt
    elif optimizer_type == "random":
        opt = RandomSearchOptimizer(seed=seed)
        opt.actual_type = "random"
        return opt
    elif optimizer_type == "bayesian":
        opt = BayesianOptimizer(random_state=seed)
        opt.actual_type = "bayesian"
        return opt
    elif optimizer_type in ("optuna", "tpe"):
        if HAS_OPTUNA:
            opt = OptunaOptimizer(seed=seed)
            opt.actual_type = "optuna"
            print(f"[Optimizer] Using Optuna TPE (v{OPTUNA_VERSION})")
        else:
            print(f"[WARNING] Optuna requested but not installed. Falling back to RandomSearch.")
            print(f"[WARNING] Install with: pip install optuna")
            opt = RandomSearchOptimizer(seed=seed)
            opt.actual_type = "random (optuna fallback)"
        return opt
    else:
        print(f"[WARNING] Unknown optimizer: {optimizer_type}, using RandomSearch")
        opt = RandomSearchOptimizer(seed=seed)
        opt.actual_type = f"random (unknown: {optimizer_type})"
        return opt


# ============================================================
# 파라미터 공간 정의 예시
# ============================================================

DEFAULT_PARAM_SPACE = {
    # RSI 관련
    "rsi_period": [10, 12, 14, 16, 20],
    "rsi_oversold": [25, 30, 35, 40],
    "rsi_overbought": [60, 65, 70, 75],

    # EMA 관련
    "ema_short": [10, 20, 30],
    "ema_long": [50, 100, 200],

    # ATR 관련
    "atr_period": [10, 14, 20],
    "sl_atr_mult": [1.5, 2.0, 2.5, 3.0],
    "tp_atr_mult": [2.0, 3.0, 4.0, 5.0],

    # 점수 관련
    "min_score": [3.0, 3.5, 4.0, 4.5, 5.0],
    "min_tf_alignment": [2, 3],
}


__all__ = [
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
    "validate_params",
    "DEFAULT_PARAM_SPACE",
]
