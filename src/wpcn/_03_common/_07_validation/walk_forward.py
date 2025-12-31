"""
Walk-Forward 분석
- 오버피팅 방지
- 롤링 윈도우 검증
- Out-of-sample 테스트
"""

from dataclasses import dataclass
from typing import List, Callable, Any, Dict
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class WalkForwardResult:
    """Walk-Forward 결과"""
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: Dict
    test_metrics: Dict
    best_params: Dict
    is_degraded: bool  # 테스트 성과가 훈련보다 크게 떨어지면 True


class WalkForwardAnalyzer:
    """
    Walk-Forward 분석기

    사용법:
    1. 데이터를 여러 기간으로 나눔 (fold)
    2. 각 fold에서:
       - 훈련 기간: 파라미터 최적화
       - 테스트 기간: 성과 검증
    3. 모든 fold의 테스트 성과 집계
    """

    def __init__(
        self,
        n_folds: int = 5,
        train_ratio: float = 0.7,
        min_trades_per_fold: int = 30,
        degradation_threshold: float = 0.5  # 50% 성과 하락 시 degraded
    ):
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.min_trades_per_fold = min_trades_per_fold
        self.degradation_threshold = degradation_threshold

    def split_data(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp"
    ) -> List[tuple]:
        """
        데이터를 train/test 기간으로 분할

        Returns:
            [(train_df, test_df, train_dates, test_dates), ...]
        """
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        n = len(df)
        fold_size = n // self.n_folds

        splits = []

        for i in range(self.n_folds):
            fold_start = i * fold_size
            fold_end = (i + 1) * fold_size if i < self.n_folds - 1 else n

            fold_data = df.iloc[fold_start:fold_end]
            train_size = int(len(fold_data) * self.train_ratio)

            train_df = fold_data.iloc[:train_size]
            test_df = fold_data.iloc[train_size:]

            train_dates = (train_df[timestamp_col].iloc[0], train_df[timestamp_col].iloc[-1])
            test_dates = (test_df[timestamp_col].iloc[0], test_df[timestamp_col].iloc[-1])

            splits.append((train_df, test_df, train_dates, test_dates))

        return splits

    def run(
        self,
        df: pd.DataFrame,
        optimize_func: Callable[[pd.DataFrame], Dict],
        backtest_func: Callable[[pd.DataFrame, Dict], Dict],
        timestamp_col: str = "timestamp"
    ) -> List[WalkForwardResult]:
        """
        Walk-Forward 분석 실행

        Args:
            df: 전체 데이터
            optimize_func: 파라미터 최적화 함수 (df -> best_params)
            backtest_func: 백테스트 함수 (df, params -> metrics)

        Returns:
            각 fold의 결과 리스트
        """
        splits = self.split_data(df, timestamp_col)
        results = []

        for i, (train_df, test_df, train_dates, test_dates) in enumerate(splits):
            print(f"Fold {i+1}/{self.n_folds}")
            print(f"  Train: {train_dates[0]} ~ {train_dates[1]}")
            print(f"  Test:  {test_dates[0]} ~ {test_dates[1]}")

            # 1. 훈련 데이터로 파라미터 최적화
            best_params = optimize_func(train_df)
            print(f"  Best params: {best_params}")

            # 2. 훈련 데이터로 백테스트
            train_metrics = backtest_func(train_df, best_params)

            # 3. 테스트 데이터로 백테스트 (같은 파라미터)
            test_metrics = backtest_func(test_df, best_params)

            # 4. 성과 저하 체크
            train_sharpe = train_metrics.get("sharpe_ratio", 0)
            test_sharpe = test_metrics.get("sharpe_ratio", 0)

            is_degraded = False
            if train_sharpe > 0:
                degradation = (train_sharpe - test_sharpe) / train_sharpe
                is_degraded = degradation > self.degradation_threshold

            print(f"  Train Sharpe: {train_sharpe:.3f}, Test Sharpe: {test_sharpe:.3f}")
            if is_degraded:
                print(f"  ⚠️ Performance degradation detected!")

            results.append(WalkForwardResult(
                fold=i + 1,
                train_start=train_dates[0],
                train_end=train_dates[1],
                test_start=test_dates[0],
                test_end=test_dates[1],
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                best_params=best_params,
                is_degraded=is_degraded
            ))

        return results

    def summarize(self, results: List[WalkForwardResult]) -> Dict:
        """결과 요약"""
        train_sharpes = [r.train_metrics.get("sharpe_ratio", 0) for r in results]
        test_sharpes = [r.test_metrics.get("sharpe_ratio", 0) for r in results]

        train_returns = [r.train_metrics.get("total_return", 0) for r in results]
        test_returns = [r.test_metrics.get("total_return", 0) for r in results]

        degraded_folds = sum(1 for r in results if r.is_degraded)

        return {
            "n_folds": len(results),
            "degraded_folds": degraded_folds,
            "avg_train_sharpe": np.mean(train_sharpes),
            "avg_test_sharpe": np.mean(test_sharpes),
            "std_test_sharpe": np.std(test_sharpes),
            "avg_train_return": np.mean(train_returns),
            "avg_test_return": np.mean(test_returns),
            "sharpe_degradation": (np.mean(train_sharpes) - np.mean(test_sharpes)) / max(np.mean(train_sharpes), 0.01),
            "is_overfit": degraded_folds > len(results) / 2,  # 절반 이상 저하 시 오버피팅
            "robustness_score": np.mean(test_sharpes) / max(np.std(test_sharpes), 0.01)  # 높을수록 안정적
        }


class RollingWalkForward(WalkForwardAnalyzer):
    """
    롤링 윈도우 Walk-Forward
    - 고정 크기 윈도우가 이동
    - 더 많은 테스트 기간 확보
    """

    def __init__(
        self,
        train_window: int = 90,   # 훈련 기간 (일)
        test_window: int = 30,    # 테스트 기간 (일)
        step_size: int = 30,      # 이동 크기 (일)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def split_data(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp"
    ) -> List[tuple]:
        """롤링 윈도우로 분할"""
        df = df.sort_values(timestamp_col).reset_index(drop=True)

        # 일 단위로 그룹화
        df['date'] = pd.to_datetime(df[timestamp_col]).dt.date
        dates = df['date'].unique()

        splits = []
        i = 0

        while i + self.train_window + self.test_window <= len(dates):
            train_dates = dates[i:i + self.train_window]
            test_dates = dates[i + self.train_window:i + self.train_window + self.test_window]

            train_df = df[df['date'].isin(train_dates)].drop(columns=['date'])
            test_df = df[df['date'].isin(test_dates)].drop(columns=['date'])

            train_range = (pd.Timestamp(train_dates[0]), pd.Timestamp(train_dates[-1]))
            test_range = (pd.Timestamp(test_dates[0]), pd.Timestamp(test_dates[-1]))

            splits.append((train_df, test_df, train_range, test_range))

            i += self.step_size

        return splits
