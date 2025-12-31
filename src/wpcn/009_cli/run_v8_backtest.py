"""
V8 백테스트 실행 스크립트
V8 Backtest Runner

핵심 구조:
- 15M: Stoch RSI 필터 (과매수/과매도 구간)
- 5M: RSI 다이버전스 + ZigZag + 진입 타이밍
- EMA 제거, Wyckoff만 사용

FIXED:
- 경로 하드코딩 제거 (환경변수/config 사용)
- sys.path hack 제거
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import logging

warnings.filterwarnings('ignore')

from wpcn.config import (
    PATHS, get_data_path, get_results_path,
    setup_logging, DEFAULT_FUTURES_SYMBOLS
)
from wpcn.execution.futures_backtest_v8 import V8BacktestEngine, V8Config

# 로깅 설정
logger = setup_logging(level="INFO")


def load_all_data(symbol: str, timeframe: str = "5m") -> pd.DataFrame:
    """
    전체 기간 데이터 로드

    FIXED: 하드코딩된 경로 대신 config 모듈 사용
    """
    data_path = get_data_path("bronze", "binance", "futures", symbol, timeframe)

    if not data_path.exists():
        logger.warning(f"Data path not found: {data_path}")
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
                logger.warning(f"Failed to load {month_file}: {e}")

    if not dfs:
        logger.warning(f"No data files found for {symbol}")
        return None

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    return df


def run_backtest_v8(symbol: str, df: pd.DataFrame, config: V8Config) -> dict:
    """V8 백테스트 실행"""
    engine = V8BacktestEngine(config)

    try:
        result = engine.run(
            df_5m=df,
            symbol=symbol,
            initial_capital=10000.0
        )
        return result
    except Exception as e:
        import traceback
        logger.error(f"Backtest failed for {symbol}: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}


def main():
    print("=" * 80)
    print("V8 BACKTEST ENGINE - CORRECT TIMEFRAME STRUCTURE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project Root: {PATHS.PROJECT_ROOT}")
    print(f"Data Dir: {PATHS.DATA_DIR}")
    print("=" * 80)
    print("\nTimeframe Structure:")
    print("  - 4H: Wyckoff structure (no EMA)")
    print("  - 1H: Wyckoff structure (Spring/Upthrust)")
    print("  - 15M: Stoch RSI FILTER (oversold/overbought) [FIXED: lookback bug]")
    print("  - 5M: RSI Divergence + ZigZag + Entry timing [FIXED: shift applied]")
    print("=" * 80)

    # 설정 - 15x 레버리지 테스트 (승률 중심)
    config = V8Config(
        leverage=15.0,  # 15x leverage test
        position_pct=0.03,  # 낮은 포지션 (3%)
        sl_atr_mult=1.5,   # SL 여유있게 (휩소 방지)
        tp_atr_mult=2.5,   # RR ~1.67:1
        min_score=5.0,      # 더 높은 점수 요구
        min_tf_alignment=3, # 최소 3개 TF 정렬 (엄격)
        min_rr_ratio=1.5,
        warmup_bars=500,
        cooldown_bars=24,   # 2시간 쿨다운 (오버트레이딩 방지)
        max_hold_bars=72,   # 6시간 최대 홀딩
        box_lookback=50,
        spring_threshold=0.005,
        rsi_period=14,
        stoch_rsi_period=14,
        div_lookback=20,
        zigzag_threshold=0.02,
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage=0.0002,
        # P2: 확률 모델 비활성화 (테스트용)
        use_probability_model=False,
        min_tp_probability=0.50,
        min_ev_r=0.05,
        min_expected_rr=0.003
    )

    results = []

    # 15x 레버리지 테스트: BTC만 먼저 실행
    TEST_SYMBOLS = ["BTC-USDT"]  # 전체: DEFAULT_FUTURES_SYMBOLS
    for symbol in TEST_SYMBOLS:
        print(f"\n{'='*60}")
        print(f"[{symbol}] Loading ALL data...")

        df = load_all_data(symbol, "5m")

        if df is None or len(df) < 2000:
            print(f"  Insufficient data, skipping")
            results.append({
                "symbol": symbol,
                "status": "skipped",
                "reason": "insufficient data"
            })
            continue

        # 데이터 기간 정보
        start_date = df.index.min()
        end_date = df.index.max()
        days = (end_date - start_date).days

        print(f"  Loaded {len(df):,} candles")
        print(f"  Period: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({days} days)")
        print(f"  Running V8 backtest...")

        result = run_backtest_v8(symbol, df, config)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            if "traceback" in result:
                print(f"  {result['traceback'][:500]}")
            results.append({
                "symbol": symbol,
                "status": "error",
                "reason": result["error"]
            })
            continue

        stats = result["stats"]

        print(f"\n  --- Results ---")
        print(f"  Return: {stats['total_return']:.2f}%")
        print(f"  Max DD: {stats['max_drawdown']:.2f}%")
        print(f"  Trades: {stats['total_trades']} (Long: {stats['long_trades']}, Short: {stats['short_trades']})")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Signals: {stats['total_signals']}")

        if 'tp_exits' in stats:
            print(f"  Exits: TP={stats['tp_exits']}, SL={stats['sl_exits']}, Timeout={stats['timeout_exits']}, Liq={stats['liquidations']}")

        results.append({
            "symbol": symbol,
            "status": "success",
            "days": days,
            "candles": len(df),
            "total_return": stats["total_return"],
            "max_drawdown": stats["max_drawdown"],
            "total_trades": stats["total_trades"],
            "long_trades": stats["long_trades"],
            "short_trades": stats["short_trades"],
            "win_rate": stats["win_rate"],
            "profit_factor": stats["profit_factor"],
            "avg_pnl": stats["avg_pnl"],
            "total_signals": stats["total_signals"],
            "tp_exits": stats.get("tp_exits", 0),
            "sl_exits": stats.get("sl_exits", 0),
            "timeout_exits": stats.get("timeout_exits", 0),
            "liquidations": stats.get("liquidations", 0)
        })

    # 요약 리포트
    print("\n" + "=" * 80)
    print("SUMMARY REPORT - V8 STRATEGY (CORRECT TF STRUCTURE)")
    print("=" * 80)

    success_results = [r for r in results if r["status"] == "success"]

    if success_results:
        df_summary = pd.DataFrame(success_results)
        df_summary = df_summary.sort_values("total_return", ascending=False)

        print("\n[Performance Ranking by Return]")
        print("-" * 100)
        print(f"{'Symbol':<12} {'Days':>6} {'Return':>10} {'MaxDD':>10} {'WinRate':>10} {'PF':>8} {'Trades':>8} {'TP/SL':>12}")
        print("-" * 100)

        for _, row in df_summary.iterrows():
            tp_sl = f"{row['tp_exits']}/{row['sl_exits']}"
            print(f"{row['symbol']:<12} {row['days']:>6} {row['total_return']:>9.2f}% {row['max_drawdown']:>9.2f}% "
                  f"{row['win_rate']:>9.1f}% {row['profit_factor']:>7.2f} {row['total_trades']:>8} {tp_sl:>12}")

        print("-" * 100)

        # 전체 통계
        avg_return = df_summary["total_return"].mean()
        avg_wr = df_summary["win_rate"].mean()
        avg_pf = df_summary["profit_factor"].mean()
        total_trades = df_summary["total_trades"].sum()
        total_tp = df_summary["tp_exits"].sum()
        total_sl = df_summary["sl_exits"].sum()
        total_timeout = df_summary["timeout_exits"].sum()
        total_liq = df_summary["liquidations"].sum()
        profitable = len(df_summary[df_summary["total_return"] > 0])

        print(f"\n[Overall Statistics]")
        print(f"  Coins Tested: {len(success_results)}")
        print(f"  Profitable: {profitable}/{len(success_results)} ({profitable/len(success_results)*100:.1f}%)")
        print(f"  Avg Return: {avg_return:.2f}%")
        print(f"  Avg Win Rate: {avg_wr:.1f}%")
        print(f"  Avg Profit Factor: {avg_pf:.2f}")
        print(f"  Total Trades: {total_trades}")
        print(f"  Exit Distribution: TP={total_tp}, SL={total_sl}, Timeout={total_timeout}, Liq={total_liq}")

        if len(df_summary) > 0:
            best = df_summary.iloc[0]
            worst = df_summary.iloc[-1]
            print(f"\n  Best: {best['symbol']} (+{best['total_return']:.2f}%, WR={best['win_rate']:.1f}%)")
            print(f"  Worst: {worst['symbol']} ({worst['total_return']:.2f}%, WR={worst['win_rate']:.1f}%)")

        # CSV 저장 (config 모듈 사용)
        csv_file = get_results_path(f"v8_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_summary.to_csv(csv_file, index=False)
        print(f"\n  Results saved to: {csv_file}")

    print("\n" + "=" * 80)
    print("V8 Backtest Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
