"""
파라미터 최적화 모듈
- Grid Search
- Bayesian Optimization
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Tuple
import itertools
import numpy as np
import pandas as pd


@dataclass
class OptimizationResult:
    """최적화 결과"""
    best_params: Dict
    best_score: float
    all_results: List[Dict]
    n_iterations: int


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
        """
        Grid Search 실행

        Args:
            param_space: {"param_name": [value1, value2, ...]}
            objective_func: 파라미터 dict -> score (높을수록 좋음)
        """
        # 모든 조합 생성
        keys = list(param_space.keys())
        values = list(param_space.values())
        combinations = list(itertools.product(*values))

        print(f"Grid Search: {len(combinations)} combinations")

        all_results = []
        best_score = float('-inf')
        best_params = None

        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))

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
                print(f"  Progress: {i+1}/{len(combinations)}, Best: {best_score:.4f}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            n_iterations=len(combinations)
        )


class RandomSearchOptimizer(ParamOptimizer):
    """
    Random Search 최적화
    - 랜덤 샘플링
    - Grid Search보다 빠름
    """

    def optimize(
        self,
        param_space: Dict[str, Tuple],
        objective_func: Callable[[Dict], float],
        n_iterations: int = 100
    ) -> OptimizationResult:
        """
        Random Search 실행

        Args:
            param_space: {"param_name": (min, max, type)}
                type: 'int', 'float', 'choice'
                choice인 경우: (option1, option2, ..., 'choice')
        """
        print(f"Random Search: {n_iterations} iterations")

        all_results = []
        best_score = float('-inf')
        best_params = None

        for i in range(n_iterations):
            params = self._sample_params(param_space)

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
                print(f"  Progress: {i+1}/{n_iterations}, Best: {best_score:.4f}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            n_iterations=n_iterations
        )

    def _sample_params(self, param_space: Dict) -> Dict:
        """파라미터 랜덤 샘플링"""
        params = {}

        for name, spec in param_space.items():
            if spec[-1] == 'choice':
                # 선택형
                options = spec[:-1]
                params[name] = np.random.choice(options)
            elif spec[-1] == 'int':
                # 정수형
                params[name] = np.random.randint(spec[0], spec[1] + 1)
            else:
                # 실수형
                params[name] = np.random.uniform(spec[0], spec[1])

        return params


class BayesianOptimizer(ParamOptimizer):
    """
    Bayesian Optimization
    - Gaussian Process 기반
    - 효율적 탐색
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def optimize(
        self,
        param_space: Dict[str, Tuple],
        objective_func: Callable[[Dict], float],
        n_iterations: int = 50
    ) -> OptimizationResult:
        """
        Bayesian Optimization 실행

        Args:
            param_space: {"param_name": (min, max)}
        """
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

        # 목적 함수 래퍼 (최소화이므로 부호 반전)
        all_results = []

        def objective(x):
            params = dict(zip(param_names, x))
            try:
                score = objective_func(params)
            except Exception as e:
                print(f"  Error: {e}")
                score = float('-inf')

            all_results.append({"params": params, "score": score})
            return -score  # 최소화

        print(f"Bayesian Optimization: {n_iterations} iterations")

        result = gp_minimize(
            objective,
            dimensions,
            n_calls=n_iterations,
            random_state=self.random_state,
            verbose=True
        )

        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            n_iterations=n_iterations
        )


# 파라미터 공간 정의 예시
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
