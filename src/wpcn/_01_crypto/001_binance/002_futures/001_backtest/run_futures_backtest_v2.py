#!/usr/bin/env python
"""
선물 백테스트 V2 스크립트
Futures Backtest V2 Script

개선사항:
- 15분봉 축적 매매 (여러 번 진입)
- 5분봉 다이버전스 조건 완화
- 6시간 시간 청산

BTC: 15x 레버리지
ETH: 5x 레버리지
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from wpcn.core.types import BacktestCosts, BacktestConfig
from wpcn.data.ccxt_fetch import fetch_ohlcv_to_parquet
from wpcn.data.loaders import load_parquet
from wpcn.engine.backtest import DEFAULT_THETA
from wpcn.execution.futures_backtest_v2 import simulate_futures_v2, FuturesConfigV2


def run_futures_backtest_v2(
    symbol: str,
    year: int,
    leverage: float,
    accumulation_pct: float = 0.03
) -> dict:
    """선물 백테스트 V2 실행"""

    exchange = "binance"
    timeframe = "5m"

    if year == 2025:
        start_date = "2025-01-01"
        end_date = "2025-12-29"
        days = 363
    else:
        start_date = f"{year}-01-01"
        end_date = f"{year + 1}-01-01"
        days = 365

    print(f"\n{'='*70}")
    print(f"선물 백테스트 V2 - {symbol} {year}년 (레버리지: {leverage}x)")
    print(f"{'='*70}")
    print(f"기간: {start_date} ~ {end_date}")
    print(f"설정: 축적 {accumulation_pct*100:.1f}%, 6시간 청산, 조건 완화")

    # 데이터 경로
    safe_symbol = symbol.replace("/", "-")
    data_path = REPO_ROOT / "data" / "raw" / f"{exchange}_{safe_symbol}_{timeframe}_{year}.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # 5분봉 데이터 다운로드
    print(f"[1/3] 5분봉 데이터 다운로드 중...")
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
    df_5m = load_parquet(str(data_path))
    print(f"  5분봉 캔들 수: {len(df_5m):,}")

    # 선물 백테스트 설정 V2
    print(f"[3/3] 선물 백테스트 V2 실행 중...")

    costs = BacktestCosts(fee_bps=4.0, slippage_bps=5.0)

    bt_cfg = BacktestConfig(
        initial_equity=1.0,
        max_hold_bars=30,
        tp1_frac=0.5,
        use_tp2=True,
        conf_min=0.50,
        edge_min=0.60,
    )

    futures_cfg = FuturesConfigV2(
        leverage=leverage,
        margin_mode='isolated',
        maintenance_margin_rate=0.005,
        funding_rate=0.0001,
        funding_interval_bars=96,  # 5분봉 기준 8시간

        # 축적 설정
        accumulation_pct_15m=accumulation_pct,
        max_accumulation_pct=0.15,  # 최대 15% 축적

        # 5분봉 설정
        scalping_pct_5m=0.02,

        # TP/SL
        tp_pct_15m=0.015,
        sl_pct_15m=0.010,
        tp_pct_5m=0.008,
        sl_pct_5m=0.005,

        # 시간 청산
        max_hold_bars_15m=72,  # 6시간
        max_hold_bars_5m=36,   # 3시간

        # 완화된 조건
        rsi_oversold=45,
        rsi_overbought=55,
        stoch_oversold=40,
        stoch_overbought=60,
        accumulation_cooldown=6,  # 30분 쿨다운
    )

    # 백테스트 실행
    equity_df, trades_df, stats = simulate_futures_v2(
        df_5m=df_5m,
        theta=DEFAULT_THETA,
        costs=costs,
        bt_cfg=bt_cfg,
        futures_cfg=futures_cfg,
        initial_equity=1.0
    )

    # 결과 출력
    print(f"\n[결과] {symbol} {year}년 ({leverage}x 레버리지)")
    print(f"  수익률: {stats['total_return']:+.2f}%")
    print(f"  MDD: {stats['max_drawdown']:.2f}%")
    print(f"  총 거래: {stats['total_trades']}회 (15m축적: {stats['trades_15m']}, 5m: {stats['trades_5m']})")
    print(f"  축적 횟수: {stats['accumulation_count']}회")
    print(f"  승률: {stats['win_rate']:.1f}%")
    print(f"  청산: 15m={stats['liquidations_15m']}회, 5m={stats['liquidations_5m']}회")
    print(f"  펀딩비: ${stats['total_funding_paid']:.4f}")

    return {
        'symbol': symbol,
        'year': year,
        'leverage': leverage,
        'total_return': stats['total_return'],
        'mdd': stats['max_drawdown'],
        'trades': stats['total_trades'],
        'trades_15m': stats['trades_15m'],
        'trades_5m': stats['trades_5m'],
        'accumulation_count': stats['accumulation_count'],
        'win_rate': stats['win_rate'],
        'liquidations': stats['liquidations_15m'] + stats['liquidations_5m'],
        'funding_paid': stats['total_funding_paid']
    }


def main():
    """BTC 15x, ETH 5x 백테스트 (2019-2025)"""

    print("\n" + "="*70)
    print("선물 백테스트 V2")
    print("개선사항:")
    print("  - 15분봉 축적 매매 (3%씩 최대 15%)")
    print("  - 5분봉 다이버전스 조건 완화")
    print("  - 6시간 시간 청산 (펀딩비 최소화)")
    print("="*70)

    results = []

    # BTC 15x 레버리지 (2019-2025)
    print("\n" + "="*70)
    print("BTC/USDT - 15x 레버리지")
    print("="*70)

    for year in [2025, 2024, 2023, 2022, 2021, 2020, 2019]:
        try:
            result = run_futures_backtest_v2(
                symbol="BTC/USDT",
                year=year,
                leverage=15.0,
                accumulation_pct=0.03
            )
            results.append(result)
        except Exception as e:
            print(f"[오류] BTC {year}년: {e}")

    # ETH 5x 레버리지 (2019-2025)
    print("\n" + "="*70)
    print("ETH/USDT - 5x 레버리지")
    print("="*70)

    for year in [2025, 2024, 2023, 2022, 2021, 2020, 2019]:
        try:
            result = run_futures_backtest_v2(
                symbol="ETH/USDT",
                year=year,
                leverage=5.0,
                accumulation_pct=0.03
            )
            results.append(result)
        except Exception as e:
            print(f"[오류] ETH {year}년: {e}")

    # 결과 요약
    print("\n" + "="*70)
    print("선물 백테스트 V2 결과 요약")
    print("="*70)

    # BTC 결과
    btc_results = [r for r in results if r['symbol'] == 'BTC/USDT']
    if btc_results:
        print("\n[BTC/USDT - 15x 레버리지]")
        print(f"{'연도':^6} | {'수익률':^12} | {'MDD':^10} | {'거래':^6} | {'축적':^6} | {'승률':^8} | {'청산':^4}")
        print("-"*70)
        for r in btc_results:
            print(f"{r['year']:^6} | {r['total_return']:+8.2f}% | {r['mdd']:>8.2f}% | {r['trades']:^6} | {r['accumulation_count']:^6} | {r['win_rate']:>6.1f}% | {r['liquidations']:^4}")

        avg_return = sum(r['total_return'] for r in btc_results) / len(btc_results)
        avg_mdd = sum(r['mdd'] for r in btc_results) / len(btc_results)
        total_liq = sum(r['liquidations'] for r in btc_results)
        print("-"*70)
        print(f"{'평균':^6} | {avg_return:+8.2f}% | {avg_mdd:>8.2f}% | {'':^6} | {'':^6} | {'':^8} | {total_liq:^4}")

    # ETH 결과
    eth_results = [r for r in results if r['symbol'] == 'ETH/USDT']
    if eth_results:
        print("\n[ETH/USDT - 5x 레버리지]")
        print(f"{'연도':^6} | {'수익률':^12} | {'MDD':^10} | {'거래':^6} | {'축적':^6} | {'승률':^8} | {'청산':^4}")
        print("-"*70)
        for r in eth_results:
            print(f"{r['year']:^6} | {r['total_return']:+8.2f}% | {r['mdd']:>8.2f}% | {r['trades']:^6} | {r['accumulation_count']:^6} | {r['win_rate']:>6.1f}% | {r['liquidations']:^4}")

        avg_return = sum(r['total_return'] for r in eth_results) / len(eth_results)
        avg_mdd = sum(r['mdd'] for r in eth_results) / len(eth_results)
        total_liq = sum(r['liquidations'] for r in eth_results)
        print("-"*70)
        print(f"{'평균':^6} | {avg_return:+8.2f}% | {avg_mdd:>8.2f}% | {'':^6} | {'':^6} | {'':^8} | {total_liq:^4}")

    print("\n" + "="*70)

    return results


if __name__ == "__main__":
    results = main()
