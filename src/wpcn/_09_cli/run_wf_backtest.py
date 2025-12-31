"""
Walk-Forward 최적화 + V8 백테스트 실행 스크립트

사용법:
    python run_wf_backtest.py                    # 기본 실행
    python run_wf_backtest.py --mode optimize    # 최적화만
    python run_wf_backtest.py --mode backtest    # 백테스트만 (저장된 파라미터 사용)
    python run_wf_backtest.py --mode full        # 최적화 + 백테스트
"""
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from wpcn.execution.futures_backtest_v8 import V8BacktestEngine, V8Config
from wpcn.008_tuning import (
    WalkForwardConfig,
    WalkForwardOptimizer,
    run_walk_forward_v8,
    V8_PARAM_SPACE,
    ParamStore,
    OptimizedParams,
    save_optimized_params,
    load_optimized_params
)


def load_data(symbol: str = "BTC-USDT", timeframe: str = "5m") -> pd.DataFrame:
    """데이터 로드"""
    data_dir = project_root / "data" / "bronze" / "binance" / "futures"
    symbol_dir = data_dir / symbol / timeframe

    if not symbol_dir.exists():
        # 대체 경로
        alt_data_dir = project_root / "data" / "binance_futures"
        if alt_data_dir.exists():
            files = list(alt_data_dir.glob(f"*{symbol}*{timeframe}*.parquet"))
            if files:
                return pd.read_parquet(files[0])

        raise FileNotFoundError(f"데이터를 찾을 수 없습니다: {symbol_dir}")

    # 모든 parquet 파일 로드
    dfs = []
    for year_dir in sorted(symbol_dir.iterdir()):
        if year_dir.is_dir():
            for pq_file in sorted(year_dir.glob("*.parquet")):
                dfs.append(pd.read_parquet(pq_file))

    if not dfs:
        raise FileNotFoundError(f"parquet 파일을 찾을 수 없습니다: {symbol_dir}")

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    print(f"[Data] {symbol} {timeframe}: {len(df):,} bars")
    print(f"  Period: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    return df


def create_backtest_func(symbol: str):
    """
    V8 백테스트 함수 생성

    파라미터를 받아서 백테스트 결과 반환
    """
    def backtest_func(df: pd.DataFrame, params: dict) -> dict:
        """백테스트 실행"""
        # V8Config 생성 (params에서 값 추출)
        config = V8Config(
            tp_pct=params.get("tp_pct", 0.015),
            sl_pct=params.get("sl_pct", 0.010),
            min_score=params.get("min_score", 4.0),
            rsi_oversold=params.get("rsi_oversold", 30),
            rsi_overbought=params.get("rsi_overbought", 70),
            # 고정 파라미터
            leverage=3.0,
            position_pct=0.10,
            use_probability_model=True,
            use_phase_accumulation=False  # 최적화 시에는 비활성화
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
            print(f"  Backtest error: {e}")
            return {"sharpe_ratio": -999}

    return backtest_func


def run_optimization(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
    """Walk-Forward 최적화 실행"""
    print("\n" + "=" * 60)
    print("Walk-Forward Optimization")
    print("=" * 60)

    config = WalkForwardConfig(
        train_weeks=12,
        test_weeks=2,
        step_weeks=2,
        optimizer_type="random",
        n_iterations=30,
        objective="sharpe"
    )

    backtest_func = create_backtest_func(symbol)

    result = run_walk_forward_v8(
        df=df,
        backtest_func=backtest_func,
        param_space=V8_PARAM_SPACE,
        config=config,
        output_dir=str(project_root / "results" / "optimization")
    )

    # 최적 파라미터 저장
    save_optimized_params(
        symbol=symbol,
        timeframe=timeframe,
        params=result.robust_params,
        train_score=result.avg_train_score,
        test_score=result.avg_test_score,
        overfit_ratio=result.overfit_ratio,
        n_folds=len(result.folds)
    )

    print(f"\n[Optimization Complete]")
    print(f"  Robust Params: {result.robust_params}")
    print(f"  Avg Train Score: {result.avg_train_score:.4f}")
    print(f"  Avg Test Score: {result.avg_test_score:.4f}")
    print(f"  Overfit Ratio: {result.overfit_ratio:.1%}")

    return result.robust_params


def run_backtest(df: pd.DataFrame, symbol: str, timeframe: str, params: dict = None):
    """V8 백테스트 실행"""
    print("\n" + "=" * 60)
    print("V8 Backtest")
    print("=" * 60)

    # 파라미터 로드 (없으면 기본값)
    if params is None:
        params = load_optimized_params(symbol, timeframe)
        if params:
            print(f"[Loaded] Optimized params: {params}")
        else:
            print("[Warning] No optimized params found, using defaults")
            params = {}

    # V8Config 생성
    config = V8Config(
        leverage=3.0,
        position_pct=0.10,
        tp_pct=params.get("tp_pct", 0.015),
        sl_pct=params.get("sl_pct", 0.010),
        min_score=params.get("min_score", 4.0),
        rsi_oversold=params.get("rsi_oversold", 30),
        rsi_overbought=params.get("rsi_overbought", 70),
        use_probability_model=True,
        use_phase_accumulation=True  # 피보나치 그리드 활성화
    )

    engine = V8BacktestEngine(config)
    result = engine.run(df, symbol=symbol, initial_capital=10000.0)

    # 결과 출력
    stats = result.get("stats", {})

    print(f"\n[Backtest Results]")
    print(f"  Return: {stats.get('total_return_pct', 0):.2f}%")
    print(f"  Max DD: {stats.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Sharpe: {stats.get('sharpe_ratio', 0):.2f}")
    print(f"  Win Rate: {stats.get('win_rate', 0):.1f}%")
    print(f"  Profit Factor: {stats.get('profit_factor', 0):.2f}")
    print(f"  Total Trades: {stats.get('total_trades', 0)}")

    # 결과 저장
    output_dir = project_root / "results" / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"v8_{symbol}_{timeframe}_{timestamp}.json"

    import json
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "symbol": symbol,
            "timeframe": timeframe,
            "params": params,
            "stats": stats,
            "timestamp": timestamp
        }, f, indent=2, default=str, ensure_ascii=False)

    print(f"\n[Saved] {result_file}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization + V8 Backtest")
    parser.add_argument("--mode", choices=["optimize", "backtest", "full"], default="full",
                        help="실행 모드: optimize(최적화만), backtest(백테스트만), full(둘 다)")
    parser.add_argument("--symbol", default="BTC-USDT", help="심볼")
    parser.add_argument("--timeframe", default="5m", help="타임프레임")

    args = parser.parse_args()

    print("=" * 60)
    print("Walk-Forward Optimization + V8 Backtest")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Symbol: {args.symbol}")
    print(f"  Timeframe: {args.timeframe}")
    print("=" * 60)

    # 데이터 로드
    try:
        df = load_data(args.symbol, args.timeframe)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    params = None

    # 최적화
    if args.mode in ["optimize", "full"]:
        params = run_optimization(df, args.symbol, args.timeframe)

    # 백테스트
    if args.mode in ["backtest", "full"]:
        run_backtest(df, args.symbol, args.timeframe, params)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
