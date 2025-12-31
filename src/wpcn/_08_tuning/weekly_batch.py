"""
주간 배치 최적화 시스템
- 매주 일요일 새벽 실행
- 최근 N주 데이터로 파라미터 최적화
- 결과를 DB에 저장
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
from pathlib import Path
import json
import pandas as pd

from .param_optimizer import GridSearchOptimizer, RandomSearchOptimizer, DEFAULT_PARAM_SPACE


class WeeklyOptimizer:
    """
    주간 배치 최적화

    사용법:
    1. 매주 일요일 새벽 스케줄러로 실행
    2. 최근 N주 데이터로 최적화
    3. 결과를 DB와 JSON에 저장
    4. 다음 주 매매에 사용
    """

    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        lookback_weeks: int = 12,  # 최근 12주 데이터 사용
        optimizer_type: str = "random",  # "grid", "random", "bayesian"
        n_iterations: int = 100,
        data_dir: str = None,
        output_dir: str = None
    ):
        self.symbols = symbols
        self.timeframes = timeframes
        self.lookback_weeks = lookback_weeks
        self.optimizer_type = optimizer_type
        self.n_iterations = n_iterations

        base_dir = Path(__file__).parent.parent.parent.parent
        self.data_dir = Path(data_dir) if data_dir else base_dir / "data" / "bronze"
        self.output_dir = Path(output_dir) if output_dir else base_dir / "results" / "optimization"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        backtest_func: Callable[[pd.DataFrame, Dict], Dict],
        param_space: Dict = None
    ) -> Dict:
        """
        주간 최적화 실행

        Args:
            backtest_func: (df, params) -> {"sharpe_ratio": ..., "total_return": ..., ...}
            param_space: 파라미터 탐색 공간
        """
        if param_space is None:
            param_space = DEFAULT_PARAM_SPACE

        week_str = datetime.now().strftime("%Y-W%W")
        print(f"{'='*60}")
        print(f"Weekly Optimization: {week_str}")
        print(f"{'='*60}")

        results = {}

        for symbol in self.symbols:
            results[symbol] = {}

            for timeframe in self.timeframes:
                print(f"\n[{symbol}] [{timeframe}]")

                # 데이터 로드
                df = self._load_data(symbol, timeframe)
                if df is None or len(df) < 1000:
                    print(f"  Insufficient data, skipping")
                    continue

                # 최적화 실행
                result = self._optimize_single(df, backtest_func, param_space)

                if result:
                    results[symbol][timeframe] = result
                    self._save_result(symbol, timeframe, week_str, result)

        # 전체 결과 저장
        self._save_summary(week_str, results)

        return results

    def _load_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """데이터 로드"""
        # bronze/binance/futures/BTC-USDT/15m/ 경로에서 로드
        symbol_path = symbol.replace("/", "-")
        data_path = self.data_dir / "binance" / "futures" / symbol_path / timeframe

        if not data_path.exists():
            return None

        # 최근 N주 데이터만 로드
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=self.lookback_weeks)

        dfs = []
        for year_dir in data_path.iterdir():
            if not year_dir.is_dir():
                continue

            for month_file in year_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(month_file)
                    dfs.append(df)
                except Exception as e:
                    print(f"  Error loading {month_file}: {e}")

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp")

        # 기간 필터
        df = df[df["timestamp"] >= start_date]

        return df

    def _optimize_single(
        self,
        df: pd.DataFrame,
        backtest_func: Callable,
        param_space: Dict
    ) -> Optional[Dict]:
        """단일 심볼/타임프레임 최적화"""

        # Optimizer 선택
        if self.optimizer_type == "grid":
            optimizer = GridSearchOptimizer()
        else:
            optimizer = RandomSearchOptimizer()

        # 목적 함수
        def objective(params: Dict) -> float:
            try:
                result = backtest_func(df, params)
                # Sharpe Ratio를 주 목표로
                return result.get("sharpe_ratio", 0)
            except Exception as e:
                return float('-inf')

        # 최적화 실행
        result = optimizer.optimize(param_space, objective, self.n_iterations)

        if result.best_params is None:
            return None

        # 최적 파라미터로 최종 백테스트
        final_result = backtest_func(df, result.best_params)

        return {
            "best_params": result.best_params,
            "best_score": result.best_score,
            "metrics": final_result,
            "n_iterations": result.n_iterations,
            "timestamp": datetime.now().isoformat()
        }

    def _save_result(
        self,
        symbol: str,
        timeframe: str,
        week: str,
        result: Dict
    ):
        """결과 저장 (JSON)"""
        symbol_clean = symbol.replace("/", "-")
        output_path = self.output_dir / f"{week}_{symbol_clean}_{timeframe}.json"

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"  Saved: {output_path}")

        # DB에도 저장
        self._save_to_db(symbol, timeframe, week, result)

    def _save_to_db(
        self,
        symbol: str,
        timeframe: str,
        week: str,
        result: Dict
    ):
        """DB에 결과 저장"""
        try:
            import importlib
            db_models = importlib.import_module("wpcn.003_common.008_database.models")
            db_repo = importlib.import_module("wpcn.003_common.008_database.repository")
            OptimizationResult = db_models.OptimizationResult
            create_tables = db_models.create_tables
            OptimizationRepository = db_repo.OptimizationRepository

            # 테이블 생성 (없으면)
            create_tables()

            opt_result = OptimizationResult(
                week=week,
                symbol=symbol,
                timeframe=timeframe,
                params=result["best_params"],
                backtest_result=result["metrics"],
                sharpe_ratio=result["metrics"].get("sharpe_ratio", 0),
                total_return=result["metrics"].get("total_return", 0),
                max_drawdown=result["metrics"].get("max_drawdown", 0),
                win_rate=result["metrics"].get("win_rate", 0),
                created_at=datetime.now()
            )

            OptimizationRepository.insert(opt_result)
            print(f"  Saved to DB")

        except Exception as e:
            print(f"  DB save failed: {e}")

    def _save_summary(self, week: str, results: Dict):
        """전체 결과 요약 저장"""
        summary_path = self.output_dir / f"{week}_summary.json"

        summary = {
            "week": week,
            "timestamp": datetime.now().isoformat(),
            "symbols": list(results.keys()),
            "results": {}
        }

        for symbol, tf_results in results.items():
            summary["results"][symbol] = {}
            for tf, result in tf_results.items():
                summary["results"][symbol][tf] = {
                    "sharpe": result["metrics"].get("sharpe_ratio", 0),
                    "return": result["metrics"].get("total_return", 0),
                    "mdd": result["metrics"].get("max_drawdown", 0)
                }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved: {summary_path}")

    def get_latest_params(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """최신 최적화 파라미터 조회"""
        try:
            import importlib
            db_repo = importlib.import_module("wpcn.003_common.008_database.repository")
            OptimizationRepository = db_repo.OptimizationRepository

            result = OptimizationRepository.get_latest(symbol, timeframe)
            if result:
                import json
                return json.loads(result["params"])

        except Exception as e:
            print(f"DB query failed: {e}")

        # DB 실패 시 파일에서 로드
        files = sorted(self.output_dir.glob(f"*_{symbol.replace('/', '-')}_{timeframe}.json"))
        if files:
            with open(files[-1]) as f:
                data = json.load(f)
                return data.get("best_params")

        return None


def run_weekly_optimization():
    """
    주간 최적화 실행 스크립트

    스케줄러에서 호출:
    - Windows Task Scheduler
    - Linux Cron: 0 3 * * 0 python -c "from weekly_batch import run_weekly_optimization; run_weekly_optimization()"
    """
    # 심볼 및 타임프레임 설정
    symbols = ["BTC/USDT", "ETH/USDT"]
    timeframes = ["15m", "1h"]

    optimizer = WeeklyOptimizer(
        symbols=symbols,
        timeframes=timeframes,
        lookback_weeks=12,
        optimizer_type="random",
        n_iterations=50
    )

    # 백테스트 함수 (실제 구현 필요)
    def backtest_func(df: pd.DataFrame, params: Dict) -> Dict:
        # TODO: 실제 백테스트 로직 연결
        # 임시 반환
        return {
            "sharpe_ratio": 0,
            "total_return": 0,
            "max_drawdown": 0,
            "win_rate": 0
        }

    results = optimizer.run(backtest_func)

    print("\n" + "="*60)
    print("Weekly Optimization Complete!")
    print("="*60)


if __name__ == "__main__":
    run_weekly_optimization()
