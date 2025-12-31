#!/usr/bin/env python
"""
장기 백테스트 스크립트 (2016년 ~ 현재)
비트코인 9년 데이터 백테스팅
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


def main():
    """BTC 2016년부터 현재까지 장기 백테스트"""

    # 설정값
    exchange = "binance"
    symbol = "BTC/USDT"
    timeframe = "15m"

    # 2019년 전체 (특정 기간 지정)
    start_date = "2019-01-01"
    end_date = "2020-01-01"
    days = 365  # 참고용

    print(f"\n{'='*70}")
    print(f"WPCN 장기 백테스팅 (2024년 전체)")
    print(f"{'='*70}")
    print(f"거래소: {exchange}")
    print(f"심볼: {symbol}")
    print(f"타임프레임: {timeframe}")
    print(f"기간: {start_date} ~ {end_date}")
    print(f"{'='*70}\n")

    # 데이터 경로
    safe_symbol = symbol.replace("/", "-")
    data_path = REPO_ROOT / "data" / "raw" / f"{exchange}_{safe_symbol}_{timeframe}_longterm.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # 데이터 다운로드 (특정 기간)
    print(f"[1/4] 데이터 다운로드 중... ({start_date} ~ {end_date})")
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
    print(f"[완료] 데이터 저장: {data_path}")

    # 데이터 로드
    print(f"\n[2/4] 데이터 로드 중...")
    df = load_parquet(str(data_path))
    print(f"[완료] 데이터 로드 완료: {len(df):,} 캔들")
    print(f"  시작: {df.index[0]}")
    print(f"  종료: {df.index[-1]}")

    # 백테스팅 설정
    print(f"\n[3/4] 백테스팅 실행 중...")
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

    print(f"MTF 활성화: {', '.join(mtf_timeframes)}")
    print(f"단타 모드: 활성화")
    print(f"Phase A/B 양방향 축적: 활성화")

    # 백테스트 실행
    engine = BacktestEngine(runs_root=str(REPO_ROOT / "runs"))
    run_id = engine.run(df, cfg, theta=DEFAULT_THETA, scalping_mode=True, use_phase_accumulation=True)
    print(f"[완료] 백테스팅 완료: {run_id}")

    # 결과 경로
    run_dir = REPO_ROOT / "runs" / run_id
    print(f"\n[4/4] 결과 분석 중...")

    # 결과 데이터 로드
    try:
        equity_df = pd.read_parquet(run_dir / "equity.parquet")
        trades_df = pd.read_parquet(run_dir / "trades.parquet")

        # 성과 지표
        initial_equity = 1.0
        final_equity = equity_df['equity'].iloc[-1]
        total_return = ((final_equity - initial_equity) / initial_equity) * 100

        # 연평균 수익률 (CAGR)
        years = days / 365
        cagr = ((final_equity / initial_equity) ** (1/years) - 1) * 100

        # 최대 낙폭
        equity_series = equity_df['equity']
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # 거래 통계
        entry_trades = trades_df[trades_df['type'] == 'ENTRY']
        accum_trades = trades_df[trades_df['type'].isin(['ACCUM_LONG', 'ACCUM_SHORT'])]
        accum_tp = trades_df[trades_df['type'].isin(['ACCUM_LONG_TP', 'ACCUM_SHORT_TP'])]
        accum_stop = trades_df[trades_df['type'].isin(['ACCUM_LONG_STOP', 'ACCUM_SHORT_STOP'])]

        print(f"\n{'='*70}")
        print(f"장기 백테스트 결과 ({days}일 / {years:.1f}년)")
        print(f"{'='*70}")
        print(f"초기 자본: $1.00")
        print(f"최종 자본: ${final_equity:.2f}")
        print(f"총 수익률: {total_return:+.2f}%")
        print(f"연평균 수익률 (CAGR): {cagr:+.2f}%")
        print(f"최대 낙폭 (MDD): {max_drawdown:.2f}%")

        print(f"\n=== 거래 통계 ===")
        print(f"메인 전략 거래: {len(entry_trades)}회")
        print(f"축적 전략 거래: {len(accum_trades)}회")

        if len(accum_tp) + len(accum_stop) > 0:
            accum_winrate = len(accum_tp) / (len(accum_tp) + len(accum_stop)) * 100
            print(f"축적 승률: {accum_winrate:.1f}%")

        # 연도별 수익률 분석
        print(f"\n=== 연도별 수익률 ===")
        equity_df['year'] = equity_df.index.year
        yearly_returns = []

        for year in sorted(equity_df['year'].unique()):
            year_data = equity_df[equity_df['year'] == year]['equity']
            if len(year_data) > 0:
                year_start = year_data.iloc[0]
                year_end = year_data.iloc[-1]
                year_return = ((year_end - year_start) / year_start) * 100
                yearly_returns.append({'year': year, 'return': year_return})
                print(f"  {year}: {year_return:+.2f}%")

        # 승률 및 손익비 계산
        if 'pnl' in trades_df.columns:
            winning = trades_df[trades_df['pnl'] > 0]
            losing = trades_df[trades_df['pnl'] < 0]

            if len(winning) + len(losing) > 0:
                winrate = len(winning) / (len(winning) + len(losing)) * 100
                avg_win = winning['pnl'].mean() if len(winning) > 0 else 0
                avg_loss = abs(losing['pnl'].mean()) if len(losing) > 0 else 0
                profit_factor = winning['pnl'].sum() / abs(losing['pnl'].sum()) if len(losing) > 0 else float('inf')

                print(f"\n=== 손익 분석 ===")
                print(f"승률: {winrate:.1f}%")
                print(f"평균 수익: {avg_win:.4f}")
                print(f"평균 손실: {avg_loss:.4f}")
                print(f"손익비 (PF): {profit_factor:.2f}")

        print(f"\n{'='*70}")
        print(f"결과 폴더: {run_dir}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"[오류] 결과 분석 중 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
