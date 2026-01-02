"""
주간 일요일 최적화 시스템 (Weekly Sunday Optimization)

매주 일요일에 최근 4주 데이터로 학습, 1주 데이터로 검증하여
전략 파라미터를 최적화합니다.

학습 기간: 4주 (28일)
검증 기간: 1주 (7일)

최적화 대상:
- TP/SL % (익절/손절 비율)
- box_L (박스 룩백 기간)
- m_freeze (박스 프리즈 기간)
- sl_atr_mult, tp_atr_mult (ATR 배수)
- min_score (최소 신호 점수)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
from pathlib import Path


@dataclass
class OptimizableParams:
    """최적화 가능한 파라미터 집합"""
    # TP/SL 파라미터
    tp_pct: float = 0.015        # 익절 % (롱: +1.5%, 숏: -1.5%)
    sl_pct: float = 0.010        # 손절 % (롱: -1.0%, 숏: +1.0%)

    # ATR 기반 TP/SL
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 2.5

    # 박스 파라미터
    box_L: int = 50              # 박스 룩백 기간
    m_freeze: int = 10           # 박스 프리즈 기간

    # 신호 필터
    min_score: float = 5.0       # 최소 신호 점수
    min_tf_alignment: int = 3    # 최소 타임프레임 정렬 수

    # 쿨다운
    cooldown_bars: int = 24      # 진입 쿨다운 (5m 기준 2시간)

    # 포지션
    position_pct: float = 0.03   # 포지션 크기 %


@dataclass
class ParamSearchSpace:
    """파라미터 탐색 공간"""
    tp_pct: List[float] = field(default_factory=lambda: [0.010, 0.012, 0.015, 0.018, 0.020])
    sl_pct: List[float] = field(default_factory=lambda: [0.008, 0.010, 0.012, 0.015])
    sl_atr_mult: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    tp_atr_mult: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0, 3.5])
    box_L: List[int] = field(default_factory=lambda: [30, 50, 70, 100])
    m_freeze: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    min_score: List[float] = field(default_factory=lambda: [3.0, 4.0, 5.0, 6.0])
    cooldown_bars: List[int] = field(default_factory=lambda: [12, 24, 36, 48])


class WeeklyOptimizer:
    """
    주간 일요일 최적화 엔진

    매주 일요일에 실행하여 최적 파라미터를 찾습니다.
    학습 기간 4주, 검증 기간 1주를 사용합니다.
    """

    def __init__(
        self,
        train_weeks: int = 4,
        validation_weeks: int = 1,
        n_candidates: int = 50,
        seed: int = 42,
        results_dir: Optional[Path] = None
    ):
        """
        Args:
            train_weeks: 학습 기간 (주 단위, 기본 4주)
            validation_weeks: 검증 기간 (주 단위, 기본 1주)
            n_candidates: 랜덤 샘플 후보 수
            seed: 랜덤 시드
            results_dir: 결과 저장 디렉토리
        """
        self.train_weeks = train_weeks
        self.validation_weeks = validation_weeks
        self.n_candidates = n_candidates
        self.rng = np.random.default_rng(seed)
        self.results_dir = results_dir or Path("results/optimization")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.search_space = ParamSearchSpace()
        self.optimization_history: List[Dict[str, Any]] = []

    def get_train_validation_split(
        self,
        df: pd.DataFrame,
        reference_date: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        학습/검증 데이터 분할

        Args:
            df: 전체 데이터 (timestamp 인덱스)
            reference_date: 기준 날짜 (기본: 현재)

        Returns:
            (train_df, validation_df)
        """
        if reference_date is None:
            reference_date = datetime.now()

        # 검증 기간: 기준 날짜 - 1주
        val_end = reference_date
        val_start = val_end - timedelta(weeks=self.validation_weeks)

        # 학습 기간: 검증 시작 - 4주
        train_end = val_start
        train_start = train_end - timedelta(weeks=self.train_weeks)

        # 데이터 분할
        train_df = df[(df.index >= train_start) & (df.index < train_end)]
        val_df = df[(df.index >= val_start) & (df.index < val_end)]

        return train_df, val_df

    def sample_params(self) -> OptimizableParams:
        """랜덤 파라미터 샘플링"""
        return OptimizableParams(
            tp_pct=float(self.rng.choice(self.search_space.tp_pct)),
            sl_pct=float(self.rng.choice(self.search_space.sl_pct)),
            sl_atr_mult=float(self.rng.choice(self.search_space.sl_atr_mult)),
            tp_atr_mult=float(self.rng.choice(self.search_space.tp_atr_mult)),
            box_L=int(self.rng.choice(self.search_space.box_L)),
            m_freeze=int(self.rng.choice(self.search_space.m_freeze)),
            min_score=float(self.rng.choice(self.search_space.min_score)),
            cooldown_bars=int(self.rng.choice(self.search_space.cooldown_bars)),
        )

    def evaluate_params(
        self,
        df: pd.DataFrame,
        params: OptimizableParams,
        backtest_func: callable
    ) -> Dict[str, float]:
        """
        파라미터 평가

        Args:
            df: 평가용 데이터
            params: 평가할 파라미터
            backtest_func: 백테스트 함수 (df, params) -> stats dict

        Returns:
            성과 지표 딕셔너리
        """
        try:
            stats = backtest_func(df, params)
            return stats
        except Exception as e:
            print(f"[Optimizer] Evaluation failed: {e}")
            return {
                "total_return": -100.0,
                "max_drawdown": 100.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0
            }

    def calculate_objective(self, stats: Dict[str, float]) -> float:
        """
        목적 함수 계산 (최대화 목표)

        = 수익률 - 최대낙폭 + 승률보너스 + 손익비보너스

        Args:
            stats: 성과 지표

        Returns:
            목적 함수 값 (높을수록 좋음)
        """
        total_return = stats.get("total_return", -100)
        max_dd = abs(stats.get("max_drawdown", 100))
        win_rate = stats.get("win_rate", 0)
        profit_factor = stats.get("profit_factor", 0)
        total_trades = stats.get("total_trades", 0)

        # 거래 수가 너무 적으면 페널티
        if total_trades < 5:
            return -100.0

        # 목적 함수: 수익률 중심, MDD 페널티, 승률/PF 보너스
        objective = (
            total_return * 1.0          # 수익률 가중치
            - max_dd * 1.5              # MDD 페널티
            + (win_rate - 50) * 0.1     # 50% 초과 승률 보너스
            + min(profit_factor, 3) * 2 # PF 보너스 (최대 3)
        )

        return objective

    def optimize(
        self,
        df: pd.DataFrame,
        backtest_func: callable,
        reference_date: Optional[datetime] = None
    ) -> Tuple[OptimizableParams, Dict[str, Any]]:
        """
        주간 최적화 실행

        Args:
            df: 전체 데이터
            backtest_func: 백테스트 함수
            reference_date: 기준 날짜

        Returns:
            (최적 파라미터, 최적화 결과)
        """
        if reference_date is None:
            reference_date = datetime.now()

        print(f"\n{'='*60}")
        print(f"[Weekly Optimizer] Starting optimization")
        print(f"  Reference Date: {reference_date.strftime('%Y-%m-%d')}")
        print(f"  Train Period: {self.train_weeks} weeks")
        print(f"  Validation Period: {self.validation_weeks} weeks")
        print(f"  Candidates: {self.n_candidates}")
        print(f"{'='*60}")

        # 데이터 분할
        train_df, val_df = self.get_train_validation_split(df, reference_date)

        if len(train_df) < 1000:
            print(f"[Warning] Insufficient training data: {len(train_df)} bars")
        if len(val_df) < 200:
            print(f"[Warning] Insufficient validation data: {len(val_df)} bars")

        print(f"  Train: {len(train_df)} bars")
        print(f"  Validation: {len(val_df)} bars")

        # 후보 평가
        candidates = []
        best_train_obj = -np.inf
        best_train_params = None

        print(f"\n[Phase 1] Training on {len(train_df)} bars...")
        for i in range(self.n_candidates):
            params = self.sample_params()
            stats = self.evaluate_params(train_df, params, backtest_func)
            obj = self.calculate_objective(stats)

            candidates.append({
                "params": params,
                "train_stats": stats,
                "train_objective": obj
            })

            if obj > best_train_obj:
                best_train_obj = obj
                best_train_params = params
                print(f"  [{i+1}/{self.n_candidates}] New best: obj={obj:.4f}, "
                      f"ret={stats.get('total_return', 0):.2f}%, "
                      f"mdd={stats.get('max_drawdown', 0):.2f}%")

        # 상위 10개 후보 검증
        candidates.sort(key=lambda x: x["train_objective"], reverse=True)
        top_candidates = candidates[:10]

        print(f"\n[Phase 2] Validating top 10 candidates on {len(val_df)} bars...")
        best_val_obj = -np.inf
        best_params = None
        best_result = None

        for i, cand in enumerate(top_candidates):
            params = cand["params"]
            val_stats = self.evaluate_params(val_df, params, backtest_func)
            val_obj = self.calculate_objective(val_stats)

            cand["val_stats"] = val_stats
            cand["val_objective"] = val_obj

            if val_obj > best_val_obj:
                best_val_obj = val_obj
                best_params = params
                best_result = {
                    "train_stats": cand["train_stats"],
                    "train_objective": cand["train_objective"],
                    "val_stats": val_stats,
                    "val_objective": val_obj,
                    "params": asdict(params)
                }
                print(f"  [{i+1}/10] New best validation: obj={val_obj:.4f}, "
                      f"ret={val_stats.get('total_return', 0):.2f}%, "
                      f"mdd={val_stats.get('max_drawdown', 0):.2f}%")

        # 결과 저장
        if best_params is not None and best_result is not None:
            self._save_result(reference_date, best_params, best_result)

        print(f"\n[Optimization Complete]")
        print(f"  Best Params: TP={best_params.tp_pct*100:.1f}%, SL={best_params.sl_pct*100:.1f}%")
        print(f"  Train: ret={best_result['train_stats'].get('total_return', 0):.2f}%")
        print(f"  Validation: ret={best_result['val_stats'].get('total_return', 0):.2f}%")

        return best_params, best_result

    def _save_result(
        self,
        reference_date: datetime,
        params: OptimizableParams,
        result: Dict[str, Any]
    ) -> None:
        """최적화 결과 저장"""
        filename = f"weekly_opt_{reference_date.strftime('%Y%m%d')}.json"
        filepath = self.results_dir / filename

        save_data = {
            "reference_date": reference_date.isoformat(),
            "params": asdict(params),
            "train_stats": result["train_stats"],
            "val_stats": result["val_stats"],
            "train_objective": result["train_objective"],
            "val_objective": result["val_objective"]
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"  Result saved to: {filepath}")
        self.optimization_history.append(save_data)

    def load_latest_params(self) -> Optional[OptimizableParams]:
        """최신 최적화 파라미터 로드"""
        files = list(self.results_dir.glob("weekly_opt_*.json"))
        if not files:
            return None

        latest_file = max(files, key=lambda f: f.stem)

        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        params_dict = data["params"]
        return OptimizableParams(**params_dict)

    def should_run_today(self) -> bool:
        """오늘이 최적화 실행일(일요일)인지 확인"""
        today = datetime.now()
        return today.weekday() == 6  # 일요일 = 6


def get_next_sunday() -> datetime:
    """다음 일요일 날짜 반환"""
    today = datetime.now()
    days_until_sunday = (6 - today.weekday()) % 7
    if days_until_sunday == 0:
        days_until_sunday = 7  # 오늘이 일요일이면 다음 주 일요일
    return today + timedelta(days=days_until_sunday)
