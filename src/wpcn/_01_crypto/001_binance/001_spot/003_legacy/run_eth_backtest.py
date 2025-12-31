#!/usr/bin/env python
"""
ETH/USDT 연도별 백테스트 스크립트 (2019년 ~ 2025년)
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from wpcn.core.types import BacktestCosts, BacktestConfig, RunConfig
from wpcn.data.ccxt_fetch import fetch_ohlcv_to_parquet
from wpcn.data.loaders import load_parquet
from wpcn.engine.backtest import BacktestEngine, DEFAULT_THETA


def run_backtest_for_year(year: int) -> dict:
    """특정 연도의 ETH 백테스트 실행"""

    exchange = "binance"
    symbol = "ETH/USDT"
    timeframe = "15m"

    if year == 2025:
        # 2025년은 현재까지
        start_date = "2025-01-01"
        end_date = "2025-12-29"
        days = 363
    else:
        start_date = f"{year}-01-01"
        end_date = f"{year + 1}-01-01"
        days = 365

    print(f"\n{'='*70}")
    print(f"ETH/USDT 백테스팅 - {year}년")
    print(f"{'='*70}")
    print(f"기간: {start_date} ~ {end_date}")

    # 데이터 경로
    safe_symbol = symbol.replace("/", "-")
    data_path = REPO_ROOT / "data" / "raw" / f"{exchange}_{safe_symbol}_{timeframe}_{year}.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # 데이터 다운로드
    print(f"[1/3] 데이터 다운로드 중...")
    fetch_ohlcv_to_parquet(
        exchange_id=exchange,
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        out_path=str(data_path),
        api_key=None,
        secret=None,
        start_date=start_date,
        end_date=end_date,
    )

    # 데이터 로드
    print(f"[2/3] 데이터 로드 중...")
    df = load_parquet(str(data_path))
    print(f"  캔들 수: {len(df):,}")

    # 백테스팅 설정
    print(f"[3/3] 백테스팅 실행 중...")
    costs = BacktestCosts(fee_bps=10.0, slippage_bps=30.0)
    bt = BacktestConfig(
        initial_equity=1.0,
        max_hold_bars=30,
        tp1_frac=0.5,
        use_tp2=True,
        conf_min=0.50,
        edge_min=0.60,
        confirm_bars=1,
        chaos_ret_atr=3.0,
        adx_len=14,
        adx_trend=15.0,
    )

    mtf_timeframes = ["1h", "4h"]

    cfg = RunConfig(
        exchange_id=exchange,
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        costs=costs,
        bt=bt,
        use_tuning=False,
        wf={},
        mtf=mtf_timeframes,
        data_path=str(data_path),
    )

    # 백테스트 실행
    engine = BacktestEngine(runs_root=str(REPO_ROOT / "runs"))
    run_id = engine.run(df, cfg, theta=DEFAULT_THETA, scalping_mode=True, use_phase_accumulation=True)

    # 결과 분석
    run_dir = REPO_ROOT / "runs" / run_id

    try:
        equity_df = pd.read_parquet(run_dir / "equity.parquet")
        trades_df = pd.read_parquet(run_dir / "trades.parquet")

        initial_equity = 1.0
        final_equity = equity_df['equity'].iloc[-1]
        total_return = ((final_equity - initial_equity) / initial_equity) * 100

        # MDD
        equity_series = equity_df['equity']
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # 거래 통계
        entry_trades = trades_df[trades_df['type'] == 'ENTRY']
        accum_trades = trades_df[trades_df['type'].isin(['ACCUM_LONG', 'ACCUM_SHORT'])]

        result = {
            'year': year,
            'total_return': total_return,
            'mdd': max_drawdown,
            'main_trades': len(entry_trades),
            'accum_trades': len(accum_trades),
            'final_equity': final_equity
        }

        print(f"\n[결과] {year}년: 수익률 {total_return:+.2f}%, MDD {max_drawdown:.2f}%")

        return result

    except Exception as e:
        print(f"[오류] 결과 분석 중 오류: {e}")
        return {
            'year': year,
            'total_return': 0,
            'mdd': 0,
            'main_trades': 0,
            'accum_trades': 0,
            'final_equity': 1.0
        }


def main():
    """ETH 2019-2025 백테스트"""

    print("\n" + "="*70)
    print("ETH/USDT 연도별 백테스트 (2019-2025)")
    print("="*70)

    results = []

    # 2025년부터 2019년까지 역순으로 실행
    for year in [2025, 2024, 2023, 2022, 2021, 2020, 2019]:
        result = run_backtest_for_year(year)
        results.append(result)

    # 결과 요약
    print("\n" + "="*70)
    print("ETH/USDT 연도별 백테스트 결과 요약")
    print("="*70)
    print(f"{'연도':^8} | {'수익률':^12} | {'MDD':^10} | {'메인거래':^10} | {'축적거래':^10}")
    print("-"*70)

    for r in results:
        print(f"{r['year']:^8} | {r['total_return']:+.2f}%{'':<5} | {r['mdd']:.2f}%{'':<4} | {r['main_trades']:^10} | {r['accum_trades']:^10}")

    # 평균 계산
    avg_return = sum(r['total_return'] for r in results) / len(results)
    avg_mdd = sum(r['mdd'] for r in results) / len(results)

    print("-"*70)
    print(f"{'평균':^8} | {avg_return:+.2f}%{'':<5} | {avg_mdd:.2f}%{'':<4} |")
    print("="*70)

    return results


if __name__ == "__main__":
    results = main()
