"""
최적화 스케줄러 (Optimization Scheduler) v2.3.1
================================================

매주 일요일 자동 최적화 실행을 위한 스케줄러입니다.
cron 또는 Windows Task Scheduler와 연동하여 사용합니다.

v2.3.1 변경사항:
- OOS_FINALIZED / DEGRADATION 알림 추가
- 0.0 truthy 버그 수정 (_fmt_pct 헬퍼)
- 버전/스텝 로그 일관성 개선

v2.3 변경사항:
- Alerting 통합 (Slack, Telegram, Discord 알림)
- 승격/롤백/에러 시 알림 전송

v2.2 변경사항:
- A/B Testing 통합 (Champion/Challenger 비교)
- 자동 승격/롤백 로직 추가
- 주간 배치 실행 순서:
  1. finalize_last_week_oos() - 지난주 OOS 확정
  2. auto_promote_or_rollback() - OOS 기반 자동 승격/롤백
  3. run_adaptive_tuning() - 새 Challenger 튜닝
  4. register_challenger() - 새 파라미터를 Challenger로 등록

v2.1 변경사항:
- OOS 실시간 추적 통합 (finalize_last_week_oos)
- ACTIVE 버전 관리 연동 (set_active)

v2 변경사항:
- run_tuning.py (15m 공식 파이프라인) 연동
- 5m 스캘핑은 --use-scalping 옵션으로 분리

사용 방법:
1. CLI 직접 실행 (spot 15m - 공식):
   python -m wpcn._08_tuning.scheduler --run-now --market spot --timeframe 15m

2. 5m 스캘핑 모드:
   python -m wpcn._08_tuning.scheduler --run-now --market spot --timeframe 5m --use-scalping

3. cron 설정 (리눅스):
   0 0 * * 0 python -m wpcn._08_tuning.scheduler --run-now --market spot --timeframe 15m

4. Windows Task Scheduler:
   매주 일요일 00:00에 실행 설정
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse
import json
import sys

# 프로젝트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 지원하는 마켓과 타임프레임
VALID_MARKETS = ("spot", "futures")
VALID_TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")


# v2.3.1: 포맷 헬퍼 (0.0도 정상 출력, None만 N/A)
def _fmt_pct(x: float) -> str:
    """퍼센트 포맷 (None이면 N/A, 0.0은 0.00%)"""
    return f"{x:.2%}" if x is not None else "N/A"


def _fmt_float(x: float) -> str:
    """소수점 포맷 (None이면 N/A)"""
    return f"{x:.4f}" if x is not None else "N/A"


class OptimizationScheduler:
    """
    주간 최적화 스케줄러 v2.3.1

    매주 일요일에 자동으로 최적화를 실행하고 결과를 저장합니다.

    v2.3.1 변경:
    - OOS_FINALIZED / DEGRADATION 알림 추가
    - 0.0 truthy 버그 수정 (_fmt_pct 헬퍼)

    v2.3 변경:
    - Alerting 통합: 승격/롤백/에러 시 Slack/Telegram/Discord 알림

    v2.2 변경:
    - A/B Testing 통합: Champion/Challenger 비교
    - 자동 승격/롤백: OOS 성능 기반
    - 주간 배치 실행 순서:
      1. finalize_last_week_oos() - 현재 ACTIVE(Challenger)의 OOS 확정
      2. auto_promote_or_rollback() - Champion vs Challenger 비교 후 자동 결정
      3. run_adaptive_tuning() - 새 파라미터 튜닝
      4. register_challenger() - 새 파라미터를 Challenger로 등록

    v2.1 변경:
    - OOS 실시간 추적 통합: 지난주 실제 성능 측정
    - ACTIVE 버전 관리: 새 파라미터를 ACTIVE로 지정

    v2 변경:
    - 기본: run_tuning.py (15m 공식 파이프라인) 사용
    - 옵션: use_scalping=True면 5m 스캘핑 모드 (weekly_optimizer)
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        symbols: Optional[List[str]] = None,
        market: str = "spot",
        timeframe: str = "15m",
        use_scalping: bool = False,
        days: int = 90,
        n_iterations: int = 30,
        train_weeks: int = 8,
        test_weeks: int = 2,
    ):
        """
        Args:
            data_dir: 데이터 디렉토리
            results_dir: 결과 저장 디렉토리
            symbols: 최적화 대상 심볼 목록
            market: 마켓 타입 ("spot" 또는 "futures")
            timeframe: 타임프레임 ("5m", "15m", "1h", "4h", "1d")
            use_scalping: True면 5m 스캘핑 전용 옵티마이저 사용 (레거시)
            days: 분석 기간 (일)
            n_iterations: 최적화 반복 횟수
            train_weeks: 학습 기간 (주)
            test_weeks: 테스트 기간 (주)
        """
        # 마켓/타임프레임 검증
        if market not in VALID_MARKETS:
            raise ValueError(f"Invalid market: {market}. Must be one of {VALID_MARKETS}")
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {VALID_TIMEFRAMES}")

        self.market = market
        self.timeframe = timeframe
        self.use_scalping = use_scalping
        self.days = days
        self.n_iterations = n_iterations
        self.train_weeks = train_weeks
        self.test_weeks = test_weeks

        self.data_dir = data_dir or PROJECT_ROOT / "data"
        self.results_dir = results_dir or PROJECT_ROOT / "results" / "optimization" / market
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.symbols = symbols or ["BTC-USDT"]
        self.status_file = self.results_dir / "scheduler_status.json"

        # v2.3: Alerting 통합
        self._alert_manager = None
        self._init_alerting()

        mode = "scalping (5m)" if use_scalping else "navigation (15m)"
        print(f"[Scheduler] Initialized: market={market}, timeframe={timeframe}, mode={mode}")

    def _init_alerting(self) -> None:
        """AlertManager 초기화"""
        try:
            from .alerting import get_alert_manager
            self._alert_manager = get_alert_manager()
            if self._alert_manager.config.has_any_channel():
                print("[Scheduler] Alerting enabled (external channels configured)")
            else:
                print("[Scheduler] Alerting enabled (console only)")
        except ImportError:
            print("[Scheduler] Alerting module not available")
            self._alert_manager = None

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

        # 일요일인지 확인 (0 = 월요일, 6 = 일요일)
        if datetime.now().weekday() != 6:
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

        v2.2: 주간 배치 실행 순서
        1. finalize_last_week_oos() - 현재 ACTIVE(Challenger)의 지난주 OOS 확정
        2. auto_promote_or_rollback() - Champion vs Challenger 비교 후 자동 결정
        3. run_adaptive_tuning() - 새 파라미터 튜닝
        4. register_challenger() - 새 파라미터를 Challenger로 등록

        Args:
            force: 강제 실행 여부

        Returns:
            실행 결과
        """
        print(f"\n{'='*60}")
        print(f"[Optimization Scheduler v2.3.1] Starting")
        print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Market: {self.market}")
        print(f"  Timeframe: {self.timeframe}")
        print(f"  Symbols: {self.symbols}")
        print(f"  Mode: {'scalping' if self.use_scalping else 'navigation (run_tuning)'}")
        print(f"  Force: {force}")
        print(f"{'='*60}")

        if not self.should_run(force):
            return {"status": "skipped", "reason": "not scheduled or already ran"}

        results = {}
        reference_date = datetime.now()

        for symbol in self.symbols:
            print(f"\n[{symbol}] Starting weekly optimization cycle...")

            try:
                # ========================================
                # Step 1: OOS 확정 (지난주 실제 성능)
                # ========================================
                oos_result = self._finalize_oos(symbol, reference_date)

                # ========================================
                # Step 2: A/B 비교 후 자동 승격/롤백
                # ========================================
                ab_result = self._auto_promote_or_rollback(symbol)

                # ========================================
                # Step 3: 새 파라미터 튜닝
                # ========================================
                if self.use_scalping:
                    # 레거시: 5m 스캘핑 전용 옵티마이저
                    tuning_result = self._run_scalping_optimization(symbol)
                else:
                    # v2: run_tuning.py (공식 15m 파이프라인)
                    tuning_result = self._run_adaptive_tuning(symbol)

                # ========================================
                # Step 4: 새 파라미터를 Challenger로 등록
                # ========================================
                if tuning_result.get("status") == "success":
                    run_id = tuning_result.get("run_id")
                    if run_id:
                        self._register_challenger(symbol, run_id)

                results[symbol] = {
                    "oos": oos_result,
                    "ab_test": ab_result,
                    "tuning": tuning_result,
                }

            except Exception as e:
                import traceback
                print(f"[{symbol}] Optimization failed: {e}")
                traceback.print_exc()
                results[symbol] = {"status": "error", "error": str(e)}

                # v2.3: 에러 알림
                if self._alert_manager:
                    self._alert_manager.notify_scheduler_error(
                        error=str(e),
                        symbol=symbol,
                        market=self.market,
                        timeframe=self.timeframe,
                    )

        # 상태 업데이트
        success = any(
            r.get("tuning", {}).get("status") == "success"
            for r in results.values()
            if isinstance(r, dict)
        )
        self._update_status(success, f"Optimized {len(self.symbols)} symbols")

        # 결과 저장
        self._save_results(results)

        # v2.3: 완료 알림
        if self._alert_manager:
            self._alert_manager.notify_scheduler_complete(
                symbols=self.symbols,
                market=self.market,
                timeframe=self.timeframe,
                results=results,
            )

        print(f"\n[Scheduler] Weekly optimization cycle complete")
        return results

    def _finalize_oos(self, symbol: str, reference_date: datetime) -> Dict[str, Any]:
        """
        Step 1: 지난주 OOS 성능 확정

        현재 ACTIVE 파라미터의 지난 주 실제 성능을 측정하고
        confidence를 업데이트합니다.

        Args:
            symbol: 심볼
            reference_date: 기준 날짜

        Returns:
            OOS 확정 결과
        """
        print(f"\n[{symbol}] Step 1: Finalizing last week OOS...")

        try:
            from .oos_tracker import OOSTracker

            tracker = OOSTracker(market=self.market)
            result = tracker.finalize_last_week_oos(
                symbol=symbol,
                timeframe=self.timeframe,
                reference_date=reference_date
            )

            if result is None:
                return {"status": "skipped", "reason": "No ACTIVE params or OOS already finalized"}

            # v2.3.1: OOS 확정 알림
            if self._alert_manager and result.status == "success":
                oos_perf = result.metrics.get("oos_performance") if result.metrics else None
                self._alert_manager.notify_oos_finalized(
                    symbol=symbol,
                    run_id=getattr(result, 'run_id', None) or "unknown",
                    oos_performance=oos_perf if oos_perf is not None else 0.0,
                    timeframe=self.timeframe,
                    market=self.market,
                    metrics=result.metrics,
                )

            # v2.3.1: 성능 저하 감지 시 알림
            if self._alert_manager and result.degradation is not None and result.degradation > 20:
                self._alert_manager.notify_degradation(
                    symbol=symbol,
                    degradation_pct=result.degradation,
                    timeframe=self.timeframe,
                    market=self.market,
                    details=result.metrics,
                )

            return {
                "status": result.status,
                "degradation": result.degradation,
                "metrics": result.metrics,
                "oos_period": f"{result.oos_start.date()} ~ {result.oos_end.date()}",
                "message": result.message,
            }

        except ImportError as e:
            print(f"  [Warning] OOS tracker not available: {e}")
            return {"status": "skipped", "reason": "OOS tracker not available"}
        except Exception as e:
            print(f"  [Error] OOS finalization failed: {e}")
            return {"status": "error", "error": str(e)}

    def _auto_promote_or_rollback(self, symbol: str) -> Dict[str, Any]:
        """
        Step 2: A/B 비교 후 자동 승격/롤백

        Champion vs Challenger OOS 성능을 비교하여:
        - Challenger가 우수하면 Champion으로 승격
        - Challenger가 현저히 나쁘면 롤백
        - 그 외에는 Challenger 유지 (추가 관찰)

        Args:
            symbol: 심볼

        Returns:
            A/B 테스트 결과
        """
        print(f"\n[{symbol}] Step 2: A/B comparison (Champion vs Challenger)...")

        try:
            from .param_versioning import get_version_manager

            manager = get_version_manager(
                symbol=symbol,
                market=self.market,
                timeframe=self.timeframe
            )

            # 자동 승격/롤백 결정
            action, comparison = manager.auto_decide()

            # v2.3.1: 0.0도 정상 출력 (_fmt_pct 사용)
            print(f"  Champion: {comparison.champion_run_id or 'None'}")
            print(f"  Challenger: {comparison.challenger_run_id or 'None'}")
            print(f"  Champion OOS: {_fmt_pct(comparison.champion_oos)}")
            print(f"  Challenger OOS: {_fmt_pct(comparison.challenger_oos)}")
            print(f"  Action: {action}")
            print(f"  Reason: {comparison.reason}")

            # v2.3.1: 승격/롤백 알림 (0.0 truthy 버그 수정)
            if self._alert_manager:
                if action == "promote":
                    self._alert_manager.notify_promotion(
                        symbol=symbol,
                        run_id=comparison.challenger_run_id,
                        timeframe=self.timeframe,
                        market=self.market,
                        old_champion_id=comparison.champion_run_id,
                        performance={
                            "Champion OOS": _fmt_pct(comparison.champion_oos),
                            "Challenger OOS": _fmt_pct(comparison.challenger_oos),
                        },
                    )
                elif action == "rollback":
                    self._alert_manager.notify_rollback(
                        symbol=symbol,
                        challenger_id=comparison.challenger_run_id,
                        champion_id=comparison.champion_run_id,
                        timeframe=self.timeframe,
                        market=self.market,
                        reason=comparison.reason,
                        performance={
                            "Champion OOS": _fmt_pct(comparison.champion_oos),
                            "Challenger OOS": _fmt_pct(comparison.challenger_oos),
                        },
                    )

            return {
                "action": action,
                "comparison": comparison.to_dict(),
                "reason": comparison.reason,
            }

        except ImportError as e:
            print(f"  [Warning] Version manager not available: {e}")
            return {"action": "skipped", "reason": "Version manager not available"}
        except Exception as e:
            print(f"  [Error] A/B comparison failed: {e}")
            return {"action": "error", "error": str(e)}

    def _register_challenger(self, symbol: str, run_id: str) -> bool:
        """
        Step 4: 새 파라미터를 Challenger로 등록

        Args:
            symbol: 심볼
            run_id: 새로 튜닝된 파라미터의 run_id

        Returns:
            성공 여부
        """
        print(f"\n[{symbol}] Step 4: Registering new params as Challenger...")

        try:
            from .param_versioning import get_version_manager

            manager = get_version_manager(
                symbol=symbol,
                market=self.market,
                timeframe=self.timeframe
            )

            success = manager.register_challenger(run_id)

            if success:
                print(f"  Challenger registered: run_id={run_id}")

                # v2.3: Challenger 등록 알림
                if self._alert_manager:
                    self._alert_manager.notify_challenger_registered(
                        symbol=symbol,
                        run_id=run_id,
                        timeframe=self.timeframe,
                        market=self.market,
                    )
            else:
                print(f"  [Warning] Failed to register Challenger")

            return success

        except ImportError as e:
            # 버전 관리 모듈이 없으면 기존 방식으로 ACTIVE 설정
            print(f"  [Warning] Version manager not available, falling back to set_active: {e}")
            return self._set_active(symbol, run_id)
        except Exception as e:
            print(f"  [Error] Failed to register Challenger: {e}")
            return False

    def _set_active(self, symbol: str, run_id: str) -> bool:
        """
        파라미터를 ACTIVE로 지정 (레거시 / 폴백용)

        Args:
            symbol: 심볼
            run_id: 파라미터의 run_id

        Returns:
            성공 여부
        """
        print(f"\n[{symbol}] Setting params as ACTIVE...")

        try:
            from .param_store import get_param_store

            store = get_param_store(market=self.market)
            success = store.set_active(symbol, self.timeframe, run_id)

            if success:
                print(f"  ACTIVE set to run_id={run_id}")
            else:
                print(f"  [Warning] Failed to set ACTIVE")

            return success

        except Exception as e:
            print(f"  [Error] Failed to set ACTIVE: {e}")
            return False

    def _run_adaptive_tuning(self, symbol: str) -> Dict[str, Any]:
        """
        v2.1: run_tuning.py 파이프라인 호출 (Step 2)

        5단계 파이프라인:
        1. Wyckoff Cycle Analysis
        2. Adaptive Parameter Space
        3. Data Loading
        4. Walk-Forward Optimization
        4.5. Sensitivity Analysis (v2.1)
        5. ParamStore 저장

        Returns:
            결과 딕셔너리 (run_id 포함)
        """
        print(f"\n[{symbol}] Step 3: Running adaptive tuning...")

        from .run_tuning import run_adaptive_tuning

        result = run_adaptive_tuning(
            symbol=symbol,
            days=self.days,
            market=self.market,
            timeframe=self.timeframe,
            n_iterations=self.n_iterations,
            train_weeks=self.train_weeks,
            test_weeks=self.test_weeks,
            output_dir=str(self.results_dir)
        )

        if result is None:
            return {"status": "error", "error": "run_adaptive_tuning returned None"}

        # run_id 추출 (ParamStore에서 생성된 ID)
        # run_id는 타임스탬프 형식: YYYYmmdd_HHMMSS
        run_id = result.get("run_id")
        if not run_id:
            # run_id가 없으면 에러 처리 (파일명 매칭 깨짐 방지)
            # v2.2: fallback으로 타임스탬프만 생성 (symbol/timeframe 제외)
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            print(f"  [Warning] run_id not returned from tuning, using fallback: {run_id}")

        return {
            "status": "success",
            "run_id": run_id,
            "params": result.get("robust_params", {}),
            "optimization": result.get("optimization", {}),
            "hints": result.get("hints", {}),
            "sensitivity": result.get("sensitivity"),
        }

    def _run_scalping_optimization(self, symbol: str) -> Dict[str, Any]:
        """
        레거시: 5m 스캘핑 전용 옵티마이저

        주의: 이 모드는 5m 타임프레임에서만 사용해야 함
        """
        from dataclasses import asdict
        from .scalping.weekly_optimizer import WeeklyOptimizer, OptimizableParams

        if self.timeframe != "5m":
            print(f"[Warning] Scalping mode is designed for 5m, but using {self.timeframe}")

        optimizer = WeeklyOptimizer(
            train_weeks=4,
            validation_weeks=1,
            n_candidates=50,
            results_dir=self.results_dir
        )

        df = self._load_data(symbol)
        if df is None or len(df) < 5000:
            return {"status": "skipped", "reason": "insufficient data"}

        backtest_func = self._create_scalping_backtest_func(symbol)

        best_params, result = optimizer.optimize(
            df,
            backtest_func,
            reference_date=datetime.now()
        )

        return {
            "status": "success",
            "params": asdict(best_params),
            "train_return": result["train_stats"].get("total_return", 0),
            "val_return": result["val_stats"].get("total_return", 0)
        }

    def _create_scalping_backtest_func(self, symbol: str):
        """
        스캘핑 모드용 백테스트 함수 (STUB)

        TODO: 실제 백테스트 엔진 연결 필요
        """
        import warnings
        from .scalping.weekly_optimizer import OptimizableParams

        warnings.warn(
            "[Scheduler] Using STUB backtest function for scalping mode! "
            "Implement with actual backtest engine before production use.",
            UserWarning
        )

        def backtest_func(df, params: OptimizableParams) -> Dict[str, float]:
            import numpy as np
            return {
                "total_return": np.random.uniform(-5, 15),
                "max_drawdown": np.random.uniform(1, 10),
                "win_rate": np.random.uniform(40, 70),
                "profit_factor": np.random.uniform(0.8, 2.5),
                "total_trades": np.random.randint(10, 100)
            }

        return backtest_func

    def _load_data(self, symbol: str):
        """
        데이터 로드 (스캘핑 모드용)

        경로: data/bronze/binance/{market}/{symbol}/{timeframe}/
        """
        import pandas as pd

        data_path = self.data_dir / "bronze" / "binance" / self.market / symbol / self.timeframe

        if not data_path.exists():
            print(f"[Scheduler] Data path not found: {data_path}")
            return None

        dfs = []
        for year_dir in data_path.iterdir():
            if not year_dir.is_dir():
                continue
            for month_file in year_dir.glob("*.parquet"):
                try:
                    df = pd.read_parquet(month_file)
                    dfs.append(df)
                except Exception as e:
                    print(f"[Scheduler] Failed to load {month_file}: {e}")
                    continue

        if not dfs:
            print(f"[Scheduler] No parquet files found in {data_path}")
            return None

        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp")
        df = df.set_index("timestamp")

        print(f"[Scheduler] Loaded {len(df)} bars from {data_path}")
        return df

    def _save_results(self, results: Dict[str, Any]) -> None:
        """결과 저장"""
        filename = f"scheduler_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"[Scheduler] Results saved to: {filepath}")

    def get_current_params(self, symbol: str, timeframe: str = None) -> Optional[Dict[str, Any]]:
        """
        현재 최적 파라미터 반환 (신뢰도 기반)

        Args:
            symbol: 심볼
            timeframe: 타임프레임 (None이면 self.timeframe)

        Returns:
            최적화된 파라미터 딕셔너리 (없으면 None)
        """
        from .param_store import get_param_store

        tf = timeframe or self.timeframe
        store = get_param_store(market=self.market)

        # 신뢰도 기반 로드 (60점 이상)
        params = store.load_reliable(
            symbol=symbol,
            timeframe=tf,
            min_confidence=60.0,
            fallback_weeks=4
        )

        return params


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(
        description="Weekly Optimization Scheduler v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 15m 네비게이션 최적화 (공식)
  python -m wpcn._08_tuning.scheduler --run-now --market spot --timeframe 15m

  # 5m 스캘핑 최적화 (레거시)
  python -m wpcn._08_tuning.scheduler --run-now --market spot --timeframe 5m --use-scalping

  # 일요일 자동 실행
  python -m wpcn._08_tuning.scheduler --check-sunday --symbols BTC-USDT ETH-USDT
        """
    )
    parser.add_argument("--run-now", action="store_true", help="Run optimization immediately")
    parser.add_argument("--check-sunday", action="store_true", help="Check if today is Sunday and run")
    parser.add_argument("--symbols", nargs="+", default=["BTC-USDT"], help="Symbols to optimize")
    parser.add_argument("--market", choices=["spot", "futures"], default="spot", help="Market type (default: spot)")
    parser.add_argument("--timeframe", choices=["5m", "15m", "1h", "4h", "1d"], default="15m", help="Timeframe (default: 15m)")
    parser.add_argument("--use-scalping", action="store_true", help="Use legacy 5m scalping optimizer instead of run_tuning")
    parser.add_argument("--days", type=int, default=90, help="Analysis period in days (default: 90)")
    parser.add_argument("--iterations", type=int, default=30, help="Optimization iterations (default: 30)")
    parser.add_argument("--train-weeks", type=int, default=8, help="Training period in weeks (default: 8)")
    parser.add_argument("--test-weeks", type=int, default=2, help="Test period in weeks (default: 2)")

    args = parser.parse_args()

    scheduler = OptimizationScheduler(
        symbols=args.symbols,
        market=args.market,
        timeframe=args.timeframe,
        use_scalping=args.use_scalping,
        days=args.days,
        n_iterations=args.iterations,
        train_weeks=args.train_weeks,
        test_weeks=args.test_weeks,
    )

    if args.run_now:
        scheduler.run(force=True)
    elif args.check_sunday:
        scheduler.run(force=False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
