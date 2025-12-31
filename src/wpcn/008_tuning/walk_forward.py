"""
Walk-Forward 최적화 모듈
- Train/Test 분할 기반 파라미터 최적화
- 과적합 방지를 위한 Out-of-Sample 검증
- V8 백테스트 엔진 연동 지원
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np
import pandas as pd

from wpcn.core.types import Theta, RunConfig
from wpcn.execution.broker_sim import simulate
from wpcn.metrics.performance import summarize_performance

# 기존 튜닝 모듈 (옵션)
try:
    from wpcn.tuning.theta_space import ThetaSpace
except ImportError:
    ThetaSpace = None

from .param_optimizer import BayesianOptimizer, RandomSearchOptimizer, OptimizationResult


# ============================================================
# 기존 Walk-Forward 함수 (하위 호환성 유지)
# ============================================================

def walk_forward_splits(n: int, train: int, test: int, embargo: int) -> List[Tuple[slice, slice]]:
    """기존 슬라이스 기반 분할"""
    splits = []
    start = 0
    while True:
        tr0 = start
        tr1 = tr0 + train
        te0 = tr1 + embargo
        te1 = te0 + test
        if te1 > n:
            break
        splits.append((slice(tr0, tr1), slice(te0, te1)))
        start = te0
    return splits


def objective(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> float:
    """기존 목적함수: mean ret - |maxdd| - turnover penalty"""
    perf = summarize_performance(equity_df, trades_df)
    mean_ret = float(equity_df["ret"].mean()) if "ret" in equity_df.columns else 0
    mdd = abs(perf.get("max_drawdown_pct", 0)) / 100.0
    tc = perf.get("trade_entries", 0)
    return mean_ret - 1.0 * mdd - 0.0005 * tc


def tune(df: pd.DataFrame, cfg: RunConfig, theta_space) -> Tuple[pd.DataFrame, Theta]:
    """기존 튜닝 함수 (하위 호환)"""
    wf = cfg.wf
    splits = walk_forward_splits(len(df), int(wf["train_bars"]), int(wf["test_bars"]), int(wf["embargo_bars"]))
    rows = []
    best_theta: Optional[Theta] = None
    best_obj = -1e9

    for fold_id, (tr_sl, te_sl) in enumerate(splits):
        train_df = df.iloc[tr_sl]
        test_df = df.iloc[te_sl]

        fold_best_theta = None
        fold_best_obj = -1e9

        for _ in range(int(wf["n_candidates"])):
            th = theta_space.sample()
            eq_tr, tr_trades, _, _ = simulate(train_df, th, cfg.costs, cfg.bt)
            obj = objective(eq_tr, tr_trades)
            if obj > fold_best_obj:
                fold_best_obj = obj
                fold_best_theta = th

        assert fold_best_theta is not None
        eq_te, te_trades, _, _ = simulate(test_df, fold_best_theta, cfg.costs, cfg.bt)
        obj_te = objective(eq_te, te_trades)
        rows.append({
            "fold_id": fold_id,
            "objective_train": fold_best_obj,
            "objective_test": obj_te,
            **asdict(fold_best_theta),
        })

        if obj_te > best_obj:
            best_obj = obj_te
            best_theta = fold_best_theta

    wf_table = pd.DataFrame(rows)
    assert best_theta is not None
    return wf_table, best_theta


# ============================================================
# V8 엔진용 Walk-Forward 최적화 (신규)
# ============================================================

@dataclass
class WalkForwardConfig:
    """Walk-Forward 설정"""
    train_weeks: int = 12           # 훈련 기간 (주)
    test_weeks: int = 2             # 테스트 기간 (주)
    step_weeks: int = 2             # 슬라이딩 스텝 (주)
    min_train_bars: int = 5000      # 최소 훈련 데이터 (봉 수)
    optimizer_type: str = "bayesian"  # "bayesian", "random"
    n_iterations: int = 50          # 최적화 반복 횟수
    objective: str = "sharpe"       # "sharpe", "calmar", "return", "sortino"


@dataclass
class FoldResult:
    """단일 Fold 결과"""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: Dict
    train_score: float
    test_score: float
    test_metrics: Dict
    is_overfit: bool  # test_score < train_score * 0.5


@dataclass
class WalkForwardResult:
    """Walk-Forward 전체 결과"""
    folds: List[FoldResult]
    avg_train_score: float
    avg_test_score: float
    overfit_ratio: float  # 과적합된 fold 비율
    best_params_by_fold: List[Dict]
    robust_params: Dict  # 가장 안정적인 파라미터 (빈도 기반)

    def to_dict(self) -> Dict:
        return {
            "n_folds": len(self.folds),
            "avg_train_score": self.avg_train_score,
            "avg_test_score": self.avg_test_score,
            "overfit_ratio": self.overfit_ratio,
            "robust_params": self.robust_params,
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "train_period": f"{f.train_start} ~ {f.train_end}",
                    "test_period": f"{f.test_start} ~ {f.test_end}",
                    "train_score": f.train_score,
                    "test_score": f.test_score,
                    "is_overfit": f.is_overfit
                }
                for f in self.folds
            ]
        }


class WalkForwardOptimizer:
    """
    Walk-Forward 최적화 (V8 엔진용)

    데이터를 Train/Test로 나누어 순차적으로 최적화하고 검증
    과적합 방지 및 실전 성능 예측
    """

    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()

    def create_folds(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp"
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int, Dict]]:
        """
        데이터를 Train/Test fold로 분할

        Returns:
            List of (train_df, test_df, fold_id, dates_dict)
        """
        cfg = self.config

        # 타임스탬프 기준 정렬
        if timestamp_col in df.columns:
            df = df.sort_values(timestamp_col).reset_index(drop=True)
            timestamps = pd.to_datetime(df[timestamp_col])
        else:
            # 인덱스가 타임스탬프인 경우
            df = df.sort_index()
            timestamps = pd.to_datetime(df.index)

        start_date = timestamps.min()
        end_date = timestamps.max()

        train_delta = timedelta(weeks=cfg.train_weeks)
        test_delta = timedelta(weeks=cfg.test_weeks)
        step_delta = timedelta(weeks=cfg.step_weeks)

        folds = []
        fold_id = 0
        current_start = start_date

        while True:
            train_end = current_start + train_delta
            test_start = train_end
            test_end = test_start + test_delta

            if test_end > end_date:
                break

            # 데이터 필터링
            if timestamp_col in df.columns:
                train_mask = (timestamps >= current_start) & (timestamps < train_end)
                test_mask = (timestamps >= test_start) & (timestamps < test_end)
            else:
                train_mask = (timestamps >= current_start) & (timestamps < train_end)
                test_mask = (timestamps >= test_start) & (timestamps < test_end)

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            # 최소 데이터 체크
            if len(train_df) >= cfg.min_train_bars and len(test_df) > 100:
                folds.append((train_df, test_df, fold_id, {
                    "train_start": current_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end
                }))
                fold_id += 1

            current_start += step_delta

        return folds

    def optimize(
        self,
        df: pd.DataFrame,
        param_space: Dict,
        backtest_func: Callable[[pd.DataFrame, Dict], Dict],
        timestamp_col: str = "timestamp"
    ) -> WalkForwardResult:
        """
        Walk-Forward 최적화 실행

        Args:
            df: 전체 데이터
            param_space: 파라미터 탐색 공간
            backtest_func: (df, params) -> {"sharpe_ratio": ..., "total_return": ..., ...}
            timestamp_col: 타임스탬프 컬럼명

        Returns:
            WalkForwardResult
        """
        cfg = self.config

        # Fold 생성
        folds = self.create_folds(df, timestamp_col)

        if len(folds) == 0:
            raise ValueError("데이터가 부족하여 Fold를 생성할 수 없습니다")

        print(f"{'='*60}")
        print(f"Walk-Forward Optimization")
        print(f"{'='*60}")
        print(f"  Train: {cfg.train_weeks}주, Test: {cfg.test_weeks}주")
        print(f"  Step: {cfg.step_weeks}주")
        print(f"  Folds: {len(folds)}")
        print(f"  Optimizer: {cfg.optimizer_type}")
        print(f"  Iterations: {cfg.n_iterations}")
        print(f"{'='*60}\n")

        # Optimizer 선택
        if cfg.optimizer_type == "bayesian":
            optimizer = BayesianOptimizer()
        else:
            optimizer = RandomSearchOptimizer()

        fold_results = []

        for train_df, test_df, fold_id, dates in folds:
            print(f"\n[Fold {fold_id + 1}/{len(folds)}]")
            print(f"  Train: {dates['train_start'].date()} ~ {dates['train_end'].date()} ({len(train_df)} bars)")
            print(f"  Test:  {dates['test_start'].date()} ~ {dates['test_end'].date()} ({len(test_df)} bars)")

            # === Train: 최적화 ===
            def train_objective(params: Dict) -> float:
                try:
                    result = backtest_func(train_df, params)
                    return self._get_score(result, cfg.objective)
                except Exception as e:
                    print(f"    Error: {e}")
                    return float('-inf')

            opt_result = optimizer.optimize(param_space, train_objective, cfg.n_iterations)
            best_params = opt_result.best_params
            train_score = opt_result.best_score

            print(f"  Train Score: {train_score:.4f}")
            print(f"  Best Params: {best_params}")

            # === Test: 검증 ===
            try:
                test_result = backtest_func(test_df, best_params)
                test_score = self._get_score(test_result, cfg.objective)
                test_metrics = test_result
            except Exception as e:
                print(f"  Test Error: {e}")
                test_score = float('-inf')
                test_metrics = {}

            print(f"  Test Score:  {test_score:.4f}")

            # 과적합 판정 (Test가 Train의 50% 미만)
            is_overfit = test_score < train_score * 0.5 if train_score > 0 else False
            if is_overfit:
                print(f"  ⚠️ OVERFIT DETECTED")

            fold_results.append(FoldResult(
                fold_id=fold_id,
                train_start=dates['train_start'],
                train_end=dates['train_end'],
                test_start=dates['test_start'],
                test_end=dates['test_end'],
                best_params=best_params,
                train_score=train_score,
                test_score=test_score,
                test_metrics=test_metrics,
                is_overfit=is_overfit
            ))

        # === 결과 집계 ===
        valid_folds = [f for f in fold_results if f.train_score > float('-inf')]

        avg_train = np.mean([f.train_score for f in valid_folds]) if valid_folds else 0
        avg_test = np.mean([f.test_score for f in valid_folds if f.test_score > float('-inf')]) if valid_folds else 0
        overfit_ratio = sum(1 for f in valid_folds if f.is_overfit) / len(valid_folds) if valid_folds else 1.0

        # Robust 파라미터: 가장 자주 선택된 값 (또는 중앙값)
        robust_params = self._find_robust_params([f.best_params for f in valid_folds])

        result = WalkForwardResult(
            folds=fold_results,
            avg_train_score=avg_train,
            avg_test_score=avg_test,
            overfit_ratio=overfit_ratio,
            best_params_by_fold=[f.best_params for f in fold_results],
            robust_params=robust_params
        )

        print(f"\n{'='*60}")
        print(f"Walk-Forward Summary")
        print(f"{'='*60}")
        print(f"  Avg Train Score: {avg_train:.4f}")
        print(f"  Avg Test Score:  {avg_test:.4f}")
        print(f"  Overfit Ratio:   {overfit_ratio:.1%}")
        print(f"  Robust Params:   {robust_params}")
        print(f"{'='*60}")

        return result

    def _get_score(self, result: Dict, objective: str) -> float:
        """목적함수 값 추출"""
        if objective == "sharpe":
            return result.get("sharpe_ratio", result.get("sharpe", 0))
        elif objective == "calmar":
            ret = result.get("total_return", 0)
            mdd = abs(result.get("max_drawdown", -1))
            return ret / mdd if mdd > 0 else 0
        elif objective == "return":
            return result.get("total_return", 0)
        elif objective == "sortino":
            return result.get("sortino_ratio", result.get("sortino", 0))
        else:
            return result.get("sharpe_ratio", 0)

    def _find_robust_params(self, params_list: List[Dict]) -> Dict:
        """
        가장 안정적인 파라미터 찾기
        - 수치형: 중앙값
        - 범주형: 최빈값
        """
        if not params_list:
            return {}

        robust = {}
        keys = params_list[0].keys() if params_list else []

        for key in keys:
            values = [p.get(key) for p in params_list if key in p]

            if not values:
                continue

            # 수치형 체크
            if all(isinstance(v, (int, float)) for v in values):
                median = np.median(values)
                # 정수형이면 반올림
                if all(isinstance(v, int) for v in values):
                    robust[key] = int(round(median))
                else:
                    robust[key] = float(median)
            else:
                # 범주형: 최빈값
                from collections import Counter
                counter = Counter(values)
                robust[key] = counter.most_common(1)[0][0]

        return robust


# V8 전용 파라미터 공간
V8_PARAM_SPACE = {
    # Wyckoff Box 파라미터
    "pivot_lr": (2, 8, 'int'),
    "box_L": (48, 192, 'int'),
    "m_freeze": (16, 64, 'int'),
    "atr_len": (10, 20, 'int'),
    "x_atr": (0.2, 0.5),
    "m_bw": (0.05, 0.15),

    # 진입/청산 파라미터
    "tp_pct": (0.01, 0.03),
    "sl_pct": (0.005, 0.015),
    "min_score": (3.0, 5.0),

    # RSI 파라미터
    "rsi_oversold": (25, 40, 'int'),
    "rsi_overbought": (60, 75, 'int'),
}


def run_walk_forward_v8(
    df: pd.DataFrame,
    backtest_func: Callable[[pd.DataFrame, Dict], Dict],
    param_space: Dict = None,
    config: WalkForwardConfig = None,
    output_dir: str = None
) -> WalkForwardResult:
    """
    V8 엔진용 Walk-Forward 최적화 실행

    Args:
        df: OHLCV 데이터 (timestamp 컬럼 필요)
        backtest_func: 백테스트 함수 (df, params) -> result_dict
        param_space: 파라미터 공간 (기본: V8_PARAM_SPACE)
        config: WalkForwardConfig (기본값 사용 시 None)
        output_dir: 결과 저장 디렉토리

    Returns:
        WalkForwardResult
    """
    config = config or WalkForwardConfig(
        train_weeks=12,
        test_weeks=2,
        step_weeks=2,
        optimizer_type="random",  # bayesian이 없으면 random 사용
        n_iterations=30,
        objective="sharpe"
    )

    optimizer = WalkForwardOptimizer(config)
    param_space = param_space or V8_PARAM_SPACE

    result = optimizer.optimize(df, param_space, backtest_func)

    # 결과 저장
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_path / f"walk_forward_{timestamp}.json"

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str, ensure_ascii=False)

        print(f"\nResult saved: {result_file}")

    return result
