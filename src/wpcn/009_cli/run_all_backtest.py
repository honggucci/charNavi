"""
전체 코인 백테스트 실행 스크립트
"""
import sys
sys.path.insert(0, r'c:\Users\gkghd\OneDrive\바탕 화면\coin_master\wpcn-backtester-cli-noflask\src')

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from wpcn.execution.futures_backtest_v3 import FuturesBacktestV3, FuturesConfigV3
from wpcn.core.types import Theta

# 데이터 경로
DATA_PATH = Path(r"c:\Users\gkghd\OneDrive\바탕 화면\coin_master\wpcn-backtester-cli-noflask\data\bronze\binance\futures")

# 코인 목록
SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "XRP-USDT", "SOL-USDT", "DOGE-USDT",
    "LINK-USDT", "ZEC-USDT", "ICP-USDT", "FIL-USDT", "TRX-USDT", "BNB-USDT"
]


def load_data(symbol: str, timeframe: str = "5m", days: int = 30) -> pd.DataFrame:
    """데이터 로드"""
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

    # 최근 N일
    end_date = df["timestamp"].max()
    start_date = end_date - timedelta(days=days)
    df = df[df["timestamp"] >= start_date]
    df = df.set_index("timestamp")

    return df


def run_backtest(symbol: str, df: pd.DataFrame, config: FuturesConfigV3, theta: Theta) -> dict:
    """백테스트 실행"""
    engine = FuturesBacktestV3(config)

    try:
        result = engine.run(
            df_5m=df,
            theta=theta,
            initial_capital=10000.0,
            save_to_db=False
        )
        return result
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 70)
    print("FULL BACKTEST - ALL COINS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 설정
    config = FuturesConfigV3(
        leverage=10.0,
        risk_per_trade=0.01,
        max_position_pct=0.2,
        max_positions=2,
        max_daily_loss_pct=0.03,
        use_dynamic_sl_tp=True,
        min_rr_ratio=1.5,
        tp_pct=0.02,
        sl_pct=0.01,
        max_hold_bars=48,
        min_score=3.5,
        entry_cooldown=6
    )

    theta = Theta(
        pivot_lr=5,
        box_L=20,
        m_freeze=3,
        atr_len=14,
        x_atr=1.5,
        m_bw=0.02,
        N_reclaim=5,
        N_fill=2,
        F_min=0.5
    )

    results = []

    for symbol in SYMBOLS:
        print(f"\n{'='*50}")
        print(f"[{symbol}] Loading data...")

        df = load_data(symbol, "5m", days=14)

        if df is None or len(df) < 500:
            print(f"  Insufficient data, skipping")
            results.append({
                "symbol": symbol,
                "status": "skipped",
                "reason": "insufficient data"
            })
            continue

        print(f"  Loaded {len(df)} candles ({df.index.min()} ~ {df.index.max()})")
        print(f"  Running backtest...")

        result = run_backtest(symbol, df, config, theta)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            results.append({
                "symbol": symbol,
                "status": "error",
                "reason": result["error"]
            })
            continue

        stats = result["stats"]
        metrics = result.get("metrics")

        print(f"\n  --- Results ---")
        print(f"  Return: {stats['total_return']:.2f}%")
        print(f"  Max DD: {stats['max_drawdown']:.2f}%")
        print(f"  Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Fees: ${stats['total_fees']:.2f}")

        if metrics:
            print(f"  Sharpe: {metrics.sharpe_ratio:.3f}")

        results.append({
            "symbol": symbol,
            "status": "success",
            "total_return": stats["total_return"],
            "max_drawdown": stats["max_drawdown"],
            "total_trades": stats["total_trades"],
            "win_rate": stats["win_rate"],
            "avg_pnl": stats["avg_pnl"],
            "total_fees": stats["total_fees"],
            "sharpe": metrics.sharpe_ratio if metrics else 0,
            "sortino": metrics.sortino_ratio if metrics else 0,
            "profit_factor": 0  # 별도 계산 필요
        })

    # 요약 리포트
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    success_results = [r for r in results if r["status"] == "success"]

    if success_results:
        df_summary = pd.DataFrame(success_results)
        df_summary = df_summary.sort_values("total_return", ascending=False)

        print("\n[Performance Ranking]")
        print("-" * 70)
        print(f"{'Symbol':<12} {'Return':>10} {'MaxDD':>10} {'WinRate':>10} {'Trades':>8} {'Sharpe':>10}")
        print("-" * 70)

        for _, row in df_summary.iterrows():
            print(f"{row['symbol']:<12} {row['total_return']:>9.2f}% {row['max_drawdown']:>9.2f}% "
                  f"{row['win_rate']:>9.1f}% {row['total_trades']:>8} {row['sharpe']:>10.3f}")

        print("-" * 70)

        # 전체 통계
        avg_return = df_summary["total_return"].mean()
        avg_wr = df_summary["win_rate"].mean()
        total_trades = df_summary["total_trades"].sum()
        profitable = len(df_summary[df_summary["total_return"] > 0])

        print(f"\n[Overall Statistics]")
        print(f"  Coins Tested: {len(success_results)}")
        print(f"  Profitable: {profitable}/{len(success_results)} ({profitable/len(success_results)*100:.1f}%)")
        print(f"  Avg Return: {avg_return:.2f}%")
        print(f"  Avg Win Rate: {avg_wr:.1f}%")
        print(f"  Total Trades: {total_trades}")

        # Best/Worst
        best = df_summary.iloc[0]
        worst = df_summary.iloc[-1]
        print(f"\n  Best: {best['symbol']} (+{best['total_return']:.2f}%)")
        print(f"  Worst: {worst['symbol']} ({worst['total_return']:.2f}%)")

    print("\n" + "=" * 70)
    print("Backtest Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
