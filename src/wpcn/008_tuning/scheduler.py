"""
최적화 스케줄러 (Optimization Scheduler)

매주 일요일 자동 최적화 실행을 위한 스케줄러입니다.
cron 또는 Windows Task Scheduler와 연동하여 사용합니다.

사용 방법:
1. CLI 직접 실행:
   python -m wpcn.tuning.scheduler --run-now

2. 일요일 체크 후 실행:
   python -m wpcn.tuning.scheduler --check-sunday

3. cron 설정 (리눅스):
   0 0 * * 0 python -m wpcn.tuning.scheduler --run-now

4. Windows Task Scheduler:
   매주 일요일 00:00에 실행 설정
"""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import json
import sys

# 프로젝트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wpcn.tuning.weekly_optimizer import WeeklyOptimizer, OptimizableParams


class OptimizationScheduler:
    """
    주간 최적화 스케줄러

    매주 일요일에 자동으로 최적화를 실행하고 결과를 저장합니다.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        symbols: Optional[list] = None
    ):
        """
        Args:
            data_dir: 데이터 디렉토리
            results_dir: 결과 저장 디렉토리
            symbols: 최적화 대상 심볼 목록
        """
        self.data_dir = data_dir or PROJECT_ROOT / "data"
        self.results_dir = results_dir or PROJECT_ROOT / "results" / "optimization"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.symbols = symbols or ["BTC-USDT", "ETH-USDT"]

        self.optimizer = WeeklyOptimizer(
            train_weeks=4,
            validation_weeks=1,
            n_candidates=50,
            results_dir=self.results_dir
        )

        self.status_file = self.results_dir / "scheduler_status.json"

    def should_run(self, force: bool = False) -> bool:
        """
        최적화 실행 여부 확인

        Args:
            force: 강제 실행 여부

        Returns:
            실행 필요 여부
        """
        if force:
            return True

        # 일요일인지 확인
        if not self.optimizer.should_run_today():
            print("[Scheduler] Today is not Sunday. Skipping.")
            return False

        # 오늘 이미 실행했는지 확인
        if self._already_ran_today():
            print("[Scheduler] Already ran today. Skipping.")
            return False

        return True

    def _already_ran_today(self) -> bool:
        """오늘 이미 실행했는지 확인"""
        if not self.status_file.exists():
            return False

        with open(self.status_file, "r", encoding="utf-8") as f:
            status = json.load(f)

        last_run = status.get("last_run_date", "")
        today = datetime.now().strftime("%Y-%m-%d")

        return last_run == today

    def _update_status(self, success: bool, message: str = "") -> None:
        """상태 파일 업데이트"""
        status = {
            "last_run_date": datetime.now().strftime("%Y-%m-%d"),
            "last_run_time": datetime.now().isoformat(),
            "success": success,
            "message": message
        }

        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)

    def run(self, force: bool = False) -> Dict[str, Any]:
        """
        최적화 실행

        Args:
            force: 강제 실행 여부

        Returns:
            실행 결과
        """
        print(f"\n{'='*60}")
        print(f"[Optimization Scheduler] Starting")
        print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Symbols: {self.symbols}")
        print(f"  Force: {force}")
        print(f"{'='*60}")

        if not self.should_run(force):
            return {"status": "skipped", "reason": "not scheduled or already ran"}

        results = {}

        for symbol in self.symbols:
            print(f"\n[{symbol}] Starting optimization...")

            try:
                # 데이터 로드 (실제 구현에서는 데이터 로더 사용)
                df = self._load_data(symbol)

                if df is None or len(df) < 5000:
                    print(f"[{symbol}] Insufficient data, skipping")
                    results[symbol] = {"status": "skipped", "reason": "insufficient data"}
                    continue

                # 백테스트 함수 (실제 구현에서는 V8BacktestEngine 사용)
                backtest_func = self._create_backtest_func(symbol)

                # 최적화 실행
                best_params, result = self.optimizer.optimize(
                    df,
                    backtest_func,
                    reference_date=datetime.now()
                )

                results[symbol] = {
                    "status": "success",
                    "params": asdict(best_params),
                    "train_return": result["train_stats"].get("total_return", 0),
                    "val_return": result["val_stats"].get("total_return", 0)
                }

            except Exception as e:
                print(f"[{symbol}] Optimization failed: {e}")
                results[symbol] = {"status": "error", "error": str(e)}

        # 상태 업데이트
        success = any(r.get("status") == "success" for r in results.values())
        self._update_status(success, f"Optimized {len(self.symbols)} symbols")

        # 결과 저장
        self._save_results(results)

        print(f"\n[Scheduler] Optimization complete")
        return results

    def _load_data(self, symbol: str):
        """
        데이터 로드 (stub - 실제 구현에서 대체)

        실제 사용 시 wpcn.data.ccxt_fetch 또는 로컬 parquet 사용
        """
        import pandas as pd

        # 데이터 경로
        data_path = self.data_dir / "bronze" / "binance" / "futures" / symbol / "5m"

        if not data_path.exists():
            return None

        dfs = []
        for year_dir in data_path.iterdir():
            if not year_dir.is_dir():
                continue
            for month_file in year_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(month_file)
                    dfs.append(df)
                except Exception:
                    continue

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        df = df.set_index("timestamp")

        return df

    def _create_backtest_func(self, symbol: str):
        """
        백테스트 함수 생성 (stub - 실제 구현에서 대체)

        실제 사용 시 V8BacktestEngine.run 사용
        """
        def backtest_func(df, params: OptimizableParams) -> Dict[str, float]:
            """간단한 백테스트 stub"""
            # 실제 구현에서는 V8BacktestEngine 사용
            # from wpcn.execution.futures_backtest_v8 import V8BacktestEngine, V8Config
            # config = V8Config(tp_atr_mult=params.tp_atr_mult, ...)
            # engine = V8BacktestEngine(config)
            # result = engine.run(df, symbol)
            # return result["stats"]

            # Stub: 랜덤 결과 반환 (테스트용)
            import numpy as np
            return {
                "total_return": np.random.uniform(-5, 15),
                "max_drawdown": np.random.uniform(1, 10),
                "win_rate": np.random.uniform(40, 70),
                "profit_factor": np.random.uniform(0.8, 2.5),
                "total_trades": np.random.randint(10, 100)
            }

        return backtest_func

    def _save_results(self, results: Dict[str, Any]) -> None:
        """결과 저장"""
        filename = f"scheduler_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"[Scheduler] Results saved to: {filepath}")

    def get_current_params(self, symbol: str) -> Optional[OptimizableParams]:
        """
        현재 최적 파라미터 반환

        Args:
            symbol: 심볼

        Returns:
            최적화된 파라미터 (없으면 기본값)
        """
        params = self.optimizer.load_latest_params()
        if params is None:
            return OptimizableParams()  # 기본값
        return params


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(description="Weekly Optimization Scheduler")
    parser.add_argument("--run-now", action="store_true", help="Run optimization immediately")
    parser.add_argument("--check-sunday", action="store_true", help="Check if today is Sunday and run")
    parser.add_argument("--symbols", nargs="+", default=["BTC-USDT"], help="Symbols to optimize")

    args = parser.parse_args()

    scheduler = OptimizationScheduler(symbols=args.symbols)

    if args.run_now:
        scheduler.run(force=True)
    elif args.check_sunday:
        scheduler.run(force=False)
    else:
        print("Usage: python -m wpcn.tuning.scheduler --run-now | --check-sunday")
        print("  --run-now: Run optimization immediately")
        print("  --check-sunday: Check if today is Sunday and run if so")


if __name__ == "__main__":
    main()
