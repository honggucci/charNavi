"""
MTF 전체 백테스트 실행 스크립트 (V4 엔진 사용)
- STRATEGY_ARCHITECTURE.md 기반 완전한 MTF 전략
- 전체 기간 데이터 사용
"""
import sys
sys.path.insert(0, r'c:\Users\gkghd\OneDrive\바탕 화면\coin_master\wpcn-backtester-cli-noflask\src')

import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from wpcn.execution.futures_backtest_v4 import MTFBacktestEngine, MTFConfig

# 데이터 경로
DATA_PATH = Path(r"c:\Users\gkghd\OneDrive\바탕 화면\coin_master\wpcn-backtester-cli-noflask\data\bronze\binance\futures")

# 코인 목록
SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "XRP-USDT", "SOL-USDT", "DOGE-USDT",
    "LINK-USDT", "ZEC-USDT", "ICP-USDT", "FIL-USDT", "TRX-USDT", "BNB-USDT"
]


def load_all_data(symbol: str, timeframe: str = "5m") -> pd.DataFrame:
    """전체 기간 데이터 로드 (days 제한 없음)"""
    data_path = DATA_PATH / symbol / timeframe

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
                pass

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    return df


def run_backtest_v4(symbol: str, df: pd.DataFrame, config: MTFConfig) -> dict:
    """V4 백테스트 실행"""
    engine = MTFBacktestEngine(config)

    try:
        result = engine.run(
            df_5m=df,
            symbol=symbol,
            initial_capital=10000.0
        )
        return result
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


def main():
    print("=" * 80)
    print("MTF FULL BACKTEST - STRATEGY_ARCHITECTURE.md")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\nStrategy Configuration:")
    print("  - 4H: Trend Filter (EMA 50/200)")
    print("  - 1H: Wyckoff Structure (Accumulation/Distribution)")
    print("  - 15M: RSI Divergence + Fibonacci")
    print("  - 5M: Entry Timing (Stoch RSI + Reversal Candles)")
    print("  - Min Score: 5.0, Min TF Alignment: 3, Min RR: 2.0")
    print("  - Look-ahead bias FIXED: Using completed candles only")
    print("=" * 80)

    # MTF 설정 (최적화 버전)
    config = MTFConfig(
        leverage=3.0,           # 3배 레버리지 (리스크 감소)
        position_pct=0.10,      # 10% 포지션 (리스크 감소)
        min_rr_ratio=2.0,       # 최소 RR 2.0 (더 엄격)
        sl_atr_mult=1.5,        # SL = ATR * 1.5 (더 타이트)
        tp_atr_mult=3.0,        # TP = ATR * 3.0 (RR 2:1)
        min_score=5.0,          # 최소 점수 5.0 (더 엄격한 신호)
        min_tf_alignment=3,     # 최소 3개 타임프레임 일치 (더 엄격)
        warmup_bars=1000,       # 워밍업 1000봉
        analysis_interval=48,   # 4시간마다 분석 (오버트레이딩 방지)
        cooldown_bars=144,      # 12시간 쿨다운 (더 보수적)
        max_hold_bars=288,      # 최대 24시간 보유 (스윙 트레이딩)
        ema_short=50,
        ema_long=200,
        box_lookback=50,
        spring_threshold=0.005,
        rsi_period=14,
        stoch_rsi_period=14,
        div_lookback=20,
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage=0.0001
    )

    results = []

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"[{symbol}] Loading ALL data...")

        df = load_all_data(symbol, "5m")

        if df is None or len(df) < 2000:  # 최소 2000봉 필요 (워밍업 1000봉 + α)
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
        print(f"  Running MTF backtest...")

        result = run_backtest_v4(symbol, df, config)

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
    print("SUMMARY REPORT - MTF STRATEGY V4")
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

        # Best/Worst
        best = df_summary.iloc[0]
        worst = df_summary.iloc[-1]
        print(f"\n  Best: {best['symbol']} (+{best['total_return']:.2f}%, WR={best['win_rate']:.1f}%)")
        print(f"  Worst: {worst['symbol']} ({worst['total_return']:.2f}%, WR={worst['win_rate']:.1f}%)")

        # CSV 저장
        output_path = Path(r"c:\Users\gkghd\OneDrive\바탕 화면\coin_master\wpcn-backtester-cli-noflask\results")
        output_path.mkdir(exist_ok=True)

        csv_file = output_path / f"mtf_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_summary.to_csv(csv_file, index=False)
        print(f"\n  Results saved to: {csv_file}")

    print("\n" + "=" * 80)
    print("MTF Backtest Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
