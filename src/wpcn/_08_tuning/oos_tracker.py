"""
OOS (Out-of-Sample) Tracker - 실시간 성능 추적
==============================================

주간 배치에서 지난 주 OOS 성능을 확정하고 confidence를 업데이트합니다.

핵심 기능:
1. ACTIVE 파라미터 추적 (실제 사용 중인 파라미터)
2. 지난 주 OOS 기간 자동 계산
3. OOS 성능 측정 및 ParamStore 업데이트
4. confidence_score 재계산

사용법:
    from wpcn._08_tuning.oos_tracker import OOSTracker

    tracker = OOSTracker(market="spot")

    # 주간 배치에서 호출
    tracker.finalize_last_week_oos(
        symbol="BTC-USDT",
        timeframe="15m",
        reference_date=datetime.now()
    )

주간 배치 실행 순서:
1. finalize_last_week_oos()  # 먼저: 지난주 OOS 확정
2. run_adaptive_tuning()      # 그 다음: 새 파라미터 튜닝
3. set_active(new_run_id)     # 마지막: 새 파라미터 ACTIVE 지정
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import json
import pandas as pd


@dataclass
class OOSResult:
    """OOS 측정 결과"""
    run_id: str
    symbol: str
    timeframe: str
    oos_start: datetime
    oos_end: datetime
    metrics: Dict[str, float]  # {return, sharpe, mdd, trades, profit_factor}
    status: str  # "success", "missing_data", "error"
    degradation: float  # oos_score / val_score (1.0이면 동일)
    message: str = ""

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "oos_start": self.oos_start.isoformat(),
            "oos_end": self.oos_end.isoformat(),
            "metrics": self.metrics,
            "status": self.status,
            "degradation": self.degradation,
            "message": self.message,
        }


class OOSTracker:
    """
    OOS (Out-of-Sample) 성능 추적기

    주간 배치에서 지난 주 실제 성능을 측정하고
    ParamStore의 confidence를 업데이트합니다.
    """

    # 최소 데이터 요구량 (봉 수)
    MIN_BARS = 500

    # OOS 성능 저하 임계값
    DEGRADATION_WARNING = 0.7   # 70% 미만이면 경고
    DEGRADATION_CRITICAL = 0.5  # 50% 미만이면 위험

    def __init__(
        self,
        market: str = "spot",
        data_dir: Optional[Path] = None,
        backtest_func: Optional[Callable] = None,
    ):
        """
        Args:
            market: 마켓 타입 ("spot" 또는 "futures")
            data_dir: 데이터 디렉토리
            backtest_func: 백테스트 함수 (없으면 기본 엔진 사용)
        """
        self.market = market
        self.data_dir = data_dir or self._get_default_data_dir()
        self._backtest_func = backtest_func

    def _get_default_data_dir(self) -> Path:
        """기본 데이터 디렉토리"""
        base = Path(__file__).parent.parent.parent.parent
        return base / "data" / "bronze" / "binance" / self.market

    def _get_backtest_func(self) -> Callable:
        """백테스트 함수 반환 (lazy load)"""
        if self._backtest_func:
            return self._backtest_func

        # 기본 백테스트 함수 생성
        from wpcn._03_common._01_core.types import Theta, BacktestConfig, BacktestCosts
        from wpcn._04_execution.broker_sim import simulate
        from wpcn._03_common._05_metrics.performance import summarize_performance

        def backtest_func(df: pd.DataFrame, params: Dict) -> Dict[str, float]:
            """기본 백테스트 함수"""
            theta = Theta(
                pivot_lr=params.get("pivot_lr", 4),
                box_L=params.get("box_L", 96),
                m_freeze=params.get("m_freeze", 32),
                atr_len=params.get("atr_len", 14),
                x_atr=params.get("x_atr", 0.35),
                m_bw=params.get("m_bw", 0.10),
                N_reclaim=params.get("N_reclaim", 3),
                N_fill=params.get("N_fill", 5),
                F_min=params.get("F_min", 0.70),
            )
            costs = BacktestCosts()
            bt_config = BacktestConfig(
                tp_pct=params.get("tp_pct", 0.015),
                sl_pct=params.get("sl_pct", 0.010),
            )

            try:
                equity_df, trades_df, _, _ = simulate(df, theta, costs, bt_config)
                perf = summarize_performance(equity_df, trades_df)

                return {
                    "total_return": perf.get("total_return_pct", 0),
                    "sharpe_ratio": perf.get("sharpe_ratio", 0),
                    "max_drawdown": perf.get("max_drawdown_pct", 0),
                    "win_rate": perf.get("win_rate", 0),
                    "profit_factor": perf.get("profit_factor", 0),
                    "total_trades": perf.get("trade_entries", 0),
                }
            except Exception as e:
                return {
                    "total_return": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                    "win_rate": 0,
                    "profit_factor": 0,
                    "total_trades": 0,
                    "_error": str(e),
                }

        self._backtest_func = backtest_func
        return backtest_func

    def get_last_week_window(
        self,
        reference_date: datetime
    ) -> Tuple[datetime, datetime]:
        """
        지난 주 OOS 기간 계산

        일요일 00:00 기준으로 [지난주 일요일, 이번주 일요일) 반환

        Args:
            reference_date: 기준 날짜 (보통 현재 시각)

        Returns:
            (oos_start, oos_end)
        """
        # 이번 주 일요일 00:00
        days_since_sunday = reference_date.weekday() + 1  # 월=0, 일=6
        if days_since_sunday == 7:  # 이미 일요일이면
            days_since_sunday = 0
        this_sunday = reference_date.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=days_since_sunday)

        # 지난주 일요일
        last_sunday = this_sunday - timedelta(weeks=1)

        return last_sunday, this_sunday

    def load_oos_data(
        self,
        symbol: str,
        timeframe: str,
        oos_start: datetime,
        oos_end: datetime
    ) -> Optional[pd.DataFrame]:
        """
        OOS 기간 데이터 로드

        Args:
            symbol: 심볼
            timeframe: 타임프레임
            oos_start: OOS 시작
            oos_end: OOS 종료

        Returns:
            DataFrame 또는 None
        """
        data_path = self.data_dir / symbol / timeframe

        if not data_path.exists():
            print(f"[OOSTracker] Data path not found: {data_path}")
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
                    print(f"[OOSTracker] Failed to load {month_file}: {e}")

        if not dfs:
            return None

        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp")

        # 타임스탬프 필터링
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        mask = (df["timestamp"] >= oos_start) & (df["timestamp"] < oos_end)
        df = df[mask].copy()

        return df if len(df) >= self.MIN_BARS else None

    def calculate_degradation(
        self,
        oos_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        metric_key: str = "sharpe_ratio"
    ) -> float:
        """
        OOS 성능 저하율 계산

        Args:
            oos_metrics: OOS 성능
            val_metrics: Validation 성능
            metric_key: 비교 지표

        Returns:
            degradation (1.0이면 동일, 0.5면 50% 저하)
        """
        oos_val = oos_metrics.get(metric_key, 0)
        val_val = val_metrics.get("score", val_metrics.get(metric_key, 0))

        if val_val <= 0:
            return 0.0

        # OOS가 음수면 심각한 저하
        if oos_val < 0:
            return max(0, 1 + oos_val / abs(val_val))

        return min(2.0, oos_val / val_val)  # 2배 이상은 2.0으로 제한

    def finalize_last_week_oos(
        self,
        symbol: str,
        timeframe: str,
        reference_date: datetime = None
    ) -> Optional[OOSResult]:
        """
        지난 주 OOS 성능 확정

        1. ACTIVE 파라미터 조회
        2. 지난 주 데이터로 백테스트
        3. ParamStore 업데이트

        Args:
            symbol: 심볼
            timeframe: 타임프레임
            reference_date: 기준 날짜 (기본: now)

        Returns:
            OOSResult 또는 None
        """
        from .param_store import get_param_store

        if reference_date is None:
            reference_date = datetime.now()

        print(f"\n[OOSTracker] Finalizing OOS for {symbol}/{timeframe}")

        store = get_param_store(market=self.market)

        # 1. ACTIVE 파라미터 조회
        active = store.get_active(symbol, timeframe)
        if not active:
            print(f"  No ACTIVE params found")
            return None

        # 이미 OOS가 채워져 있으면 스킵 (단, oos_pending=True면 재시도)
        # v2.1.1: missing_data로 스킵된 경우 oos_pending=True로 재시도 허용
        if active.performance.oos is not None:
            # oos_pending 체크: True면 재시도 허용
            oos_pending = getattr(active.confidence, 'oos_pending', None)
            if oos_pending is None:
                # confidence에 oos_pending이 없으면 confidence dict에서 직접 확인
                conf_dict = active.confidence.to_dict() if hasattr(active.confidence, 'to_dict') else {}
                oos_pending = conf_dict.get('oos_pending', False)

            if not oos_pending:
                print(f"  OOS already finalized for run_id={active.run_id}")
                return None
            else:
                print(f"  OOS pending retry (previous: missing_data)")

        run_id = active.run_id if hasattr(active, 'run_id') else f"{symbol}_{timeframe}"

        # 2. OOS 기간 계산
        oos_start, oos_end = self.get_last_week_window(reference_date)
        print(f"  OOS Period: {oos_start.date()} ~ {oos_end.date()}")

        # 3. 데이터 로드
        df = self.load_oos_data(symbol, timeframe, oos_start, oos_end)
        if df is None:
            result = OOSResult(
                run_id=run_id,
                symbol=symbol,
                timeframe=timeframe,
                oos_start=oos_start,
                oos_end=oos_end,
                metrics={},
                status="missing_data",
                degradation=0.0,
                message=f"Insufficient data for OOS period"
            )
            # 데이터 부족 시 confidence 하락
            store.update_oos(
                symbol=symbol,
                timeframe=timeframe,
                oos_metrics={"status": "missing_data"},
                oos_period=(oos_start, oos_end)
            )
            print(f"  [Warning] {result.message}")
            return result

        print(f"  Loaded {len(df)} bars")

        # 4. ACTIVE 파라미터를 봉 단위로 변환
        # CRITICAL: load_reliable_as_bars가 아닌 ACTIVE.params를 직접 변환해야 함
        from .param_schema import convert_params_to_bars

        if not active.params:
            print(f"  [Error] ACTIVE params is empty")
            return None

        # ACTIVE 파라미터 (분 단위) → 봉 단위 변환
        params_bars = convert_params_to_bars(active.params, timeframe)
        print(f"  Using ACTIVE params: box_L={params_bars.get('box_L')}")

        # 5. 백테스트 실행
        backtest_func = self._get_backtest_func()
        try:
            oos_metrics = backtest_func(df, params_bars)
            print(f"  OOS Sharpe: {oos_metrics.get('sharpe_ratio', 0):.4f}")
            print(f"  OOS Return: {oos_metrics.get('total_return', 0):.2f}%")
        except Exception as e:
            result = OOSResult(
                run_id=run_id,
                symbol=symbol,
                timeframe=timeframe,
                oos_start=oos_start,
                oos_end=oos_end,
                metrics={},
                status="error",
                degradation=0.0,
                message=str(e)
            )
            print(f"  [Error] Backtest failed: {e}")
            return result

        # 6. Degradation 계산
        val_metrics = active.performance.val if active.performance else {}
        degradation = self.calculate_degradation(oos_metrics, val_metrics)

        status = "success"
        message = ""
        if degradation < self.DEGRADATION_CRITICAL:
            status = "critical"
            message = f"OOS degradation critical: {degradation:.1%}"
        elif degradation < self.DEGRADATION_WARNING:
            status = "warning"
            message = f"OOS degradation warning: {degradation:.1%}"

        print(f"  Degradation: {degradation:.1%} ({status})")

        # 7. ParamStore 업데이트
        store.update_oos(
            symbol=symbol,
            timeframe=timeframe,
            oos_metrics=oos_metrics,
            oos_period=(oos_start, oos_end),
            degradation=degradation
        )

        result = OOSResult(
            run_id=run_id,
            symbol=symbol,
            timeframe=timeframe,
            oos_start=oos_start,
            oos_end=oos_end,
            metrics=oos_metrics,
            status=status,
            degradation=degradation,
            message=message
        )

        print(f"  OOS finalized successfully")
        return result

    def batch_finalize(
        self,
        symbols: list[str],
        timeframe: str,
        reference_date: datetime = None
    ) -> Dict[str, OOSResult]:
        """
        여러 심볼의 OOS 일괄 확정

        Args:
            symbols: 심볼 목록
            timeframe: 타임프레임
            reference_date: 기준 날짜

        Returns:
            {symbol: OOSResult}
        """
        results = {}
        for symbol in symbols:
            try:
                result = self.finalize_last_week_oos(
                    symbol=symbol,
                    timeframe=timeframe,
                    reference_date=reference_date
                )
                if result:
                    results[symbol] = result
            except Exception as e:
                print(f"[OOSTracker] Error for {symbol}: {e}")

        return results


def compute_oos_adjusted_confidence(
    base_confidence: float,
    oos_metrics: Dict[str, float],
    degradation: float
) -> float:
    """
    OOS 결과를 반영한 confidence 점수 계산

    Args:
        base_confidence: 기존 confidence (0~100)
        oos_metrics: OOS 성능 메트릭
        degradation: 성능 저하율

    Returns:
        조정된 confidence (0~100)
    """
    # OOS 가중치: 30점
    oos_weight = 30

    # degradation을 점수로 변환 (1.0 = 만점, 0.5 = 절반)
    oos_score = min(1.0, degradation) * oos_weight

    # 기존 confidence에서 OOS 비중만큼 재분배
    # (기존은 oos_performance가 10점이었으므로, 20점 추가 확보)
    adjusted_base = base_confidence * 0.7  # 70%로 축소

    return adjusted_base + oos_score
