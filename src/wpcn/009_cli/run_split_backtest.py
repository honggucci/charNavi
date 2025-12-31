"""
분할 백테스트: 최적화 구간 / 테스트 구간 분리

- 2019-01-01 ~ 2021-04-30: 파라미터 최적화 (Walk-Forward)
- 2021-05-01 ~ 현재: Out-of-Sample 백테스트

사용법:
    python run_split_backtest.py
"""
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import importlib
from wpcn.execution.futures_backtest_v8 import V8BacktestEngine, V8Config

# 008_tuning은 숫자로 시작해서 importlib 사용
tuning_module = importlib.import_module("wpcn.008_tuning")
WalkForwardConfig = tuning_module.WalkForwardConfig
WalkForwardOptimizer = tuning_module.WalkForwardOptimizer
V8_PARAM_SPACE = tuning_module.V8_PARAM_SPACE
save_optimized_params = tuning_module.save_optimized_params


# ============================================================
# 설정
# ============================================================
SYMBOL = "BTC-USDT"
TIMEFRAME = "5m"
SPLIT_DATE = "2021-05-01"  # 이 날짜부터 테스트

# 최적화 설정
TRAIN_WEEKS = 24  # 6개월 훈련
TEST_WEEKS = 8    # 2개월 테스트 (폴드 수 감소)
N_ITERATIONS = 5  # 빠른 테스트


def load_all_data() -> pd.DataFrame:
    """전체 데이터 로드"""
    data_dir = project_root / "data" / "bronze" / "binance" / "futures" / SYMBOL / TIMEFRAME

    if not data_dir.exists():
        # 대체 경로
        alt_paths = [
            project_root / "data" / "binance_futures",
            project_root / "data",
        ]
        for alt in alt_paths:
            if alt.exists():
                files = list(alt.glob(f"**/*{SYMBOL}*{TIMEFRAME}*.parquet"))
                if files:
                    dfs = [pd.read_parquet(f) for f in files]
                    df = pd.concat(dfs, ignore_index=True)
                    df = df.sort_values("timestamp").drop_duplicates("timestamp")
                    return df
        raise FileNotFoundError(f"데이터를 찾을 수 없습니다")

    # 모든 parquet 파일 로드
    dfs = []
    for year_dir in sorted(data_dir.iterdir()):
        if year_dir.is_dir():
            for pq_file in sorted(year_dir.glob("*.parquet")):
                dfs.append(pd.read_parquet(pq_file))

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df


def split_data(df: pd.DataFrame, split_date: str):
    """데이터 분할"""
    split_ts = pd.Timestamp(split_date)

    df_optimize = df[df["timestamp"] < split_ts].copy()
    df_test = df[df["timestamp"] >= split_ts].copy()

    return df_optimize, df_test


def create_backtest_func(symbol: str):
    """백테스트 함수 생성"""
    def backtest_func(df: pd.DataFrame, params: dict) -> dict:
        # FIXED: Walk-Forward 슬라이싱 후 인덱스 복원
        # V8BacktestEngine._resample()이 DatetimeIndex를 요구함
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
            # timestamp 컬럼이 없고 인덱스도 DatetimeIndex가 아니면 에러
            raise ValueError("DataFrame must have 'timestamp' column or DatetimeIndex")

        config = V8Config(
            leverage=3.0,
            position_pct=0.10,
            sl_atr_mult=params.get("sl_atr_mult", 1.5),
            tp_atr_mult=params.get("tp_atr_mult", 3.0),
            min_score=params.get("min_score", 4.0),
            rsi_period=params.get("rsi_period", 14),
            stoch_rsi_period=params.get("stoch_rsi_period", 14),
            use_probability_model=True,
            use_phase_accumulation=False  # 최적화 시 비활성화 (속도)
        )

        engine = V8BacktestEngine(config)

        try:
            result = engine.run(df, symbol=symbol, initial_capital=10000.0)
            stats = result.get("stats", {})

            return {
                "sharpe_ratio": stats.get("sharpe_ratio", 0),
                "total_return": stats.get("total_return_pct", 0) / 100,
                "max_drawdown": abs(stats.get("max_drawdown_pct", 0)) / 100,
                "win_rate": stats.get("win_rate", 0) / 100,
                "profit_factor": stats.get("profit_factor", 0),
                "n_trades": stats.get("total_trades", 0)
            }
        except Exception as e:
            print(f"  Error: {e}")
            return {"sharpe_ratio": -999}

    return backtest_func


# V8Config에 맞는 파라미터 공간
V8_PARAM_SPACE_ACTUAL = {
    # ATR 기반 SL/TP
    "sl_atr_mult": (1.0, 2.5),
    "tp_atr_mult": (2.0, 5.0),

    # 신호 조건
    "min_score": (3.0, 5.0),

    # RSI/Stoch RSI
    "rsi_period": (10, 20, 'int'),
    "stoch_rsi_period": (10, 20, 'int'),
}


def run_optimization(df_optimize: pd.DataFrame) -> dict:
    """Walk-Forward 최적화"""
    print("\n" + "=" * 70)
    print("PHASE 1: Walk-Forward Optimization")
    print("=" * 70)
    print(f"  Period: {df_optimize['timestamp'].min()} ~ {df_optimize['timestamp'].max()}")
    print(f"  Bars: {len(df_optimize):,}")

    config = WalkForwardConfig(
        train_weeks=TRAIN_WEEKS,
        test_weeks=TEST_WEEKS,
        step_weeks=8,  # 8주 스텝 (폴드 수 감소)
        optimizer_type="random",
        n_iterations=N_ITERATIONS,
        objective="sharpe"
    )

    optimizer = WalkForwardOptimizer(config)
    backtest_func = create_backtest_func(SYMBOL)

    result = optimizer.optimize(
        df_optimize,
        V8_PARAM_SPACE_ACTUAL,
        backtest_func,
        timestamp_col="timestamp"
    )

    # 저장
    save_optimized_params(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        params=result.robust_params,
        train_score=result.avg_train_score,
        test_score=result.avg_test_score,
        overfit_ratio=result.overfit_ratio,
        n_folds=len(result.folds),
        data_period=f"~{SPLIT_DATE}"
    )

    print(f"\n[Optimization Result]")
    print(f"  Robust Params: {result.robust_params}")
    print(f"  Avg Train Score: {result.avg_train_score:.4f}")
    print(f"  Avg Test Score: {result.avg_test_score:.4f}")
    print(f"  Overfit Ratio: {result.overfit_ratio:.1%}")

    return result.robust_params


def run_oos_backtest(df_test: pd.DataFrame, params: dict):
    """Out-of-Sample 백테스트"""
    print("\n" + "=" * 70)
    print("PHASE 2: Out-of-Sample Backtest")
    print("=" * 70)

    # 기간 정보 출력 (인덱스 설정 전)
    if 'timestamp' in df_test.columns:
        print(f"  Period: {df_test['timestamp'].min()} ~ {df_test['timestamp'].max()}")
    else:
        print(f"  Period: {df_test.index.min()} ~ {df_test.index.max()}")
    print(f"  Bars: {len(df_test):,}")
    print(f"  Params: {params}")

    # FIXED: DatetimeIndex 설정
    if 'timestamp' in df_test.columns:
        df_test = df_test.set_index('timestamp')

    config = V8Config(
        leverage=3.0,
        position_pct=0.10,
        sl_atr_mult=params.get("sl_atr_mult", 1.5),
        tp_atr_mult=params.get("tp_atr_mult", 3.0),
        min_score=params.get("min_score", 4.0),
        rsi_period=params.get("rsi_period", 14),
        stoch_rsi_period=params.get("stoch_rsi_period", 14),
        use_probability_model=True,
        use_phase_accumulation=True  # 피보나치 그리드 활성화
    )

    engine = V8BacktestEngine(config)
    result = engine.run(df_test, symbol=SYMBOL, initial_capital=10000.0)

    stats = result.get("stats", {})

    print(f"\n" + "=" * 70)
    print("OUT-OF-SAMPLE RESULTS (2021-05-01 ~ Present)")
    print("=" * 70)
    print(f"  Return:        {stats.get('total_return_pct', 0):>10.2f}%")
    print(f"  Max Drawdown:  {stats.get('max_drawdown_pct', 0):>10.2f}%")
    print(f"  Sharpe Ratio:  {stats.get('sharpe_ratio', 0):>10.2f}")
    print(f"  Win Rate:      {stats.get('win_rate', 0):>10.1f}%")
    print(f"  Profit Factor: {stats.get('profit_factor', 0):>10.2f}")
    print(f"  Total Trades:  {stats.get('total_trades', 0):>10d}")
    print(f"  Long Trades:   {stats.get('long_trades', 0):>10d}")
    print(f"  Short Trades:  {stats.get('short_trades', 0):>10d}")
    print("=" * 70)

    # 결과 저장
    output_dir = project_root / "results" / "split_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    import json
    result_file = output_dir / f"oos_result_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "split_date": SPLIT_DATE,
            "optimization_period": f"~{SPLIT_DATE}",
            "test_period": f"{SPLIT_DATE}~",
            "params": params,
            "stats": stats
        }, f, indent=2, default=str, ensure_ascii=False)

    print(f"\n[Saved] {result_file}")

    return result


def main():
    print("=" * 70)
    print("Split Backtest: Optimization + Out-of-Sample Test")
    print("=" * 70)
    print(f"  Symbol: {SYMBOL}")
    print(f"  Timeframe: {TIMEFRAME}")
    print(f"  Split Date: {SPLIT_DATE}")
    print("=" * 70)

    # 1. 데이터 로드
    print("\n[Loading Data...]")
    df = load_all_data()
    print(f"  Total: {len(df):,} bars")
    print(f"  Period: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    # 2. 데이터 분할
    df_optimize, df_test = split_data(df, SPLIT_DATE)
    print(f"\n[Data Split]")
    print(f"  Optimization: {len(df_optimize):,} bars ({df_optimize['timestamp'].min()} ~ {df_optimize['timestamp'].max()})")
    print(f"  Test (OOS):   {len(df_test):,} bars ({df_test['timestamp'].min()} ~ {df_test['timestamp'].max()})")

    # 3. 최적화
    params = run_optimization(df_optimize)

    # 4. Out-of-Sample 백테스트
    run_oos_backtest(df_test, params)

    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
