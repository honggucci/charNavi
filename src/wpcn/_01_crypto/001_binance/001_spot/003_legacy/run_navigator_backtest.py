#!/usr/bin/env python
"""
차트 네비게이터 백테스팅 스크립트
Chart Navigator Backtesting Script

확률론적 와이코프 스나이퍼 시스템 1년치 백테스팅
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# 경고 억제
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
pd.options.mode.chained_assignment = None

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from wpcn.core.types import BacktestCosts
from wpcn.data.ccxt_fetch import fetch_ohlcv_to_parquet
from wpcn.data.loaders import load_parquet
from wpcn.navigation.chart_navigator import ChartNavigator
from wpcn.execution.futures_broker import FuturesBroker


def run_navigator_backtest(
    symbol: str,
    exchange: str = "binance",
    timeframe: str = "15m",
    days: int = 365,
    leverage: float = 10.0,
    initial_balance: float = 10000.0
):
    """
    단일 심볼 차트 네비게이터 백테스팅

    Args:
        symbol: 거래 심볼 (예: "BTC/USDT")
        exchange: 거래소
        timeframe: 타임프레임
        days: 백테스팅 기간 (일)
        leverage: 레버리지
        initial_balance: 초기 자본
    """
    print(f"\n{'='*70}")
    print(f"차트 네비게이터 백테스팅: {symbol}")
    print(f"{'='*70}")
    print(f"거래소: {exchange}")
    print(f"타임프레임: {timeframe}")
    print(f"기간: {days}일")
    print(f"레버리지: {leverage}x")
    print(f"초기 자본: ${initial_balance:,.0f}")
    print(f"{'='*70}\n")

    # 데이터 경로
    safe_symbol = symbol.replace("/", "-")
    data_path = REPO_ROOT / "data" / "raw" / f"{exchange}_{safe_symbol}_{timeframe}_1y.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # 데이터 다운로드
    print("[1/5] 데이터 다운로드 중...")
    try:
        fetch_ohlcv_to_parquet(
            exchange_id=exchange,
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            out_path=str(data_path),
            api_key=None,
            secret=None,
        )
        print(f"[완료] 데이터 저장: {data_path}")
    except Exception as e:
        print(f"[오류] 데이터 다운로드 실패: {e}")
        return None

    # 데이터 로드
    print("\n[2/5] 데이터 로드 중...")
    df = load_parquet(str(data_path))
    print(f"[완료] 데이터 로드: {len(df)} 캔들")
    print(f"기간: {df.index[0]} ~ {df.index[-1]}")

    # 차트 네비게이터 초기화
    print("\n[3/5] 차트 네비게이터 초기화...")
    navigator = ChartNavigator(
        timeframes=['15m', '1h', '4h', '1d'],
        confluence_tolerance=0.4,
        min_confluence=2
    )

    # 선물 브로커 초기화
    costs = BacktestCosts(fee_bps=4.0, slippage_bps=5.0)  # 선물: 0.04% 수수료
    broker = FuturesBroker(
        initial_balance=initial_balance,
        leverage=leverage,
        margin_mode='isolated',
        maintenance_margin_rate=0.005,
        funding_rate=0.0001,
        max_position_pct=0.3,  # 최대 30% 포지션
        costs=costs
    )

    # 백테스팅 실행
    print("\n[4/5] 백테스팅 실행 중...")

    # 워밍업 기간 (피보나치 계산용)
    warmup = 200
    total_bars = len(df)

    signals_generated = 0
    trades_executed = 0

    # 쿨다운 추적
    last_trade_bar = -50
    cooldown_bars = 16  # 4시간 쿨다운 (15분봉 기준)
    analysis_interval = 16  # 4시간마다 분석 (성능 최적화)

    # 캐시된 분석 결과
    cached_report = None
    last_analysis_bar = 0

    for i in range(warmup, total_bars):
        t = df.index[i]
        o, h, l, c = df.iloc[i][['open', 'high', 'low', 'close']]

        # 바 처리 (기존 포지션 체크)
        event = broker.process_bar(t, i, o, h, l, c, max_hold_bars=96)  # 24시간 최대

        if event:
            trades_executed += 1

        # 포지션 없고 쿨다운 지났을 때만 신호 분석
        if broker.position is None and (i - last_trade_bar) >= cooldown_bars:
            try:
                # 네비게이션 분석 (매 16봉마다 = 4시간마다)
                if i % analysis_interval == 0:
                    current_df = df.iloc[:i+1]
                    cached_report = navigator.analyze(current_df)
                    signals_generated += 1
                    last_analysis_bar = i

                # 캐시된 리포트가 있으면 매매 신호 처리
                if cached_report and (i - last_analysis_bar) < analysis_interval * 2:
                    action = cached_report.recommended_action

                    if action in ['LONG_NOW', 'LONG_READY'] and cached_report.overall_confidence > 0.55:
                        # 롱 진입
                        entry_price = cached_report.entry_price or c
                        sl = cached_report.stop_loss or entry_price * 0.985
                        tp = cached_report.take_profit or entry_price * 1.025

                        success = broker.open_position(
                            time=t,
                            bar_idx=i,
                            price=entry_price,
                            side='long',
                            position_pct=min(0.2, cached_report.position_size_pct),
                            stop_loss=sl,
                            take_profit=tp,
                            event=f"NAV_{action}"
                        )

                        if success:
                            trades_executed += 1
                            last_trade_bar = i
                            cached_report = None  # 진입 후 리포트 무효화

                    elif action in ['SHORT_NOW', 'SHORT_READY'] and cached_report.overall_confidence > 0.55:
                        # 숏 진입
                        entry_price = cached_report.entry_price or c
                        sl = cached_report.stop_loss or entry_price * 1.015
                        tp = cached_report.take_profit or entry_price * 0.975

                        success = broker.open_position(
                            time=t,
                            bar_idx=i,
                            price=entry_price,
                            side='short',
                            position_pct=min(0.2, cached_report.position_size_pct),
                            stop_loss=sl,
                            take_profit=tp,
                            event=f"NAV_{action}"
                        )

                        if success:
                            trades_executed += 1
                            last_trade_bar = i
                            cached_report = None

                    elif action == 'ACCUMULATE' and cached_report.sc_prediction:
                        # 분할 매수 (SC 근처)
                        sc_price = cached_report.sc_prediction['price']
                        distance_pct = (c - sc_price) / c * 100

                        if distance_pct < 1.0:  # SC에서 1% 이내
                            sl = sc_price * 0.985
                            tp = c * 1.02

                            success = broker.open_position(
                                time=t,
                                bar_idx=i,
                                price=c,
                                side='long',
                                position_pct=0.05,  # 5% 분할 진입
                                stop_loss=sl,
                                take_profit=tp,
                                event="NAV_ACCUMULATE"
                            )

                            if success:
                                trades_executed += 1
                                last_trade_bar = i

                    elif action == 'DISTRIBUTE' and cached_report.bc_prediction:
                        # 분할 매도 (BC 근처)
                        bc_price = cached_report.bc_prediction['price']
                        distance_pct = (bc_price - c) / c * 100

                        if distance_pct < 1.0:
                            sl = bc_price * 1.015
                            tp = c * 0.98

                            success = broker.open_position(
                                time=t,
                                bar_idx=i,
                                price=c,
                                side='short',
                                position_pct=0.05,
                                stop_loss=sl,
                                take_profit=tp,
                                event="NAV_DISTRIBUTE"
                            )

                            if success:
                                trades_executed += 1
                                last_trade_bar = i

            except Exception as e:
                # 분석 오류 무시 (데이터 부족 등)
                pass

        # 진행률 표시 (5000봉마다)
        if i % 5000 == 0:
            progress = (i - warmup) / (total_bars - warmup) * 100
            current_equity = broker.get_equity(c)
            print(f"  진행: {progress:.1f}% | 자산: ${current_equity:,.0f} | 거래: {trades_executed}")

    # 최종 포지션 청산
    if broker.position is not None:
        broker.close_position(df.index[-1], df.iloc[-1]['close'], 'TIME', 'FINAL_CLOSE')

    # 결과 분석
    print("\n[5/5] 결과 분석 중...")

    stats = broker.get_statistics()
    equity_df = broker.get_equity_df()
    trades_df = broker.get_trades_df()

    # 결과 저장
    results_dir = REPO_ROOT / "results" / "navigator" / safe_symbol
    results_dir.mkdir(parents=True, exist_ok=True)

    if len(equity_df) > 0:
        equity_df.to_parquet(results_dir / "equity.parquet")
    if len(trades_df) > 0:
        trades_df.to_parquet(results_dir / "trades.parquet")

    # 결과 출력
    print(f"\n{'='*70}")
    print(f"백테스팅 결과: {symbol}")
    print(f"{'='*70}")
    print(f"초기 자본: ${stats.get('initial_balance', initial_balance):,.0f}")
    print(f"최종 자본: ${stats.get('final_balance', broker.balance):,.0f}")
    print(f"총 수익률: {stats.get('total_return', 0):+.2f}%")
    print(f"최대 낙폭: {stats.get('max_drawdown', 0):.2f}%")
    print(f"")
    print(f"총 거래: {stats.get('total_trades', 0)}회")
    print(f"승리: {stats.get('wins', 0)}회")
    print(f"패배: {stats.get('losses', 0)}회")
    print(f"승률: {stats.get('win_rate', 0):.1f}%")
    print(f"평균 수익: +{stats.get('avg_win_pct', 0):.2f}%")
    print(f"평균 손실: -{stats.get('avg_loss_pct', 0):.2f}%")
    print(f"손익비: {stats.get('profit_factor', 0):.2f}")
    print(f"청산 횟수: {stats.get('liquidations', 0)}회")
    print(f"레버리지: {leverage}x")
    print(f"{'='*70}")

    return {
        'symbol': symbol,
        'stats': stats,
        'equity_df': equity_df,
        'trades_df': trades_df
    }


def run_multi_symbol_backtest():
    """5개 코인 1년치 백테스팅"""

    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "TRX/USDT", "BNB/USDT"]

    print(f"\n{'#'*70}")
    print(f"# 확률론적 와이코프 스나이퍼 - 차트 네비게이터 백테스팅")
    print(f"# 5개 코인 1년치 선물 거래 시뮬레이션")
    print(f"{'#'*70}")
    print(f"심볼: {', '.join(symbols)}")
    print(f"타임프레임: 15m")
    print(f"기간: 365일")
    print(f"레버리지: 10x")
    print(f"{'#'*70}\n")

    results = {}

    for symbol in symbols:
        result = run_navigator_backtest(
            symbol=symbol,
            exchange="binance",
            timeframe="15m",
            days=365,
            leverage=10.0,
            initial_balance=10000.0
        )

        if result:
            results[symbol] = result

    # 종합 결과 출력
    print(f"\n{'='*80}")
    print(f"종합 결과")
    print(f"{'='*80}")
    print(f"{'심볼':<12} {'수익률':>12} {'MDD':>10} {'거래':>8} {'승률':>10} {'손익비':>10} {'청산':>6}")
    print(f"{'-'*80}")

    total_return = 0
    total_trades = 0

    for symbol, result in results.items():
        stats = result['stats']
        ret = stats.get('total_return', 0)
        mdd = stats.get('max_drawdown', 0)
        trades = stats.get('total_trades', 0)
        winrate = stats.get('win_rate', 0)
        pf = stats.get('profit_factor', 0)
        liq = stats.get('liquidations', 0)

        print(f"{symbol:<12} {ret:>+11.2f}% {mdd:>9.2f}% {trades:>8d} {winrate:>9.1f}% {pf:>10.2f} {liq:>6d}")

        total_return += ret
        total_trades += trades

    print(f"{'-'*80}")
    print(f"{'합계':<12} {total_return:>+11.2f}%")
    print(f"{'평균':<12} {total_return/len(results):>+11.2f}%")
    print(f"{'총 거래':<12} {total_trades:>8d}")
    print(f"{'='*80}\n")

    # 종합 분석
    print(f"\n{'='*80}")
    print(f"분석 및 시사점")
    print(f"{'='*80}")

    avg_return = total_return / len(results)
    if avg_return > 50:
        print(f"[+] 평균 수익률 {avg_return:.1f}%로 매우 우수합니다.")
    elif avg_return > 20:
        print(f"[+] 평균 수익률 {avg_return:.1f}%로 양호합니다.")
    elif avg_return > 0:
        print(f"[=] 평균 수익률 {avg_return:.1f}%로 보통입니다.")
    else:
        print(f"[-] 평균 수익률 {avg_return:.1f}%로 개선이 필요합니다.")

    # 청산 분석
    total_liq = sum(r['stats'].get('liquidations', 0) for r in results.values())
    if total_liq == 0:
        print(f"[+] 청산 0회로 리스크 관리가 우수합니다.")
    elif total_liq < 5:
        print(f"[=] 청산 {total_liq}회로 적절한 수준입니다.")
    else:
        print(f"[-] 청산 {total_liq}회로 리스크 관리 개선이 필요합니다.")

    print(f"\n[전략 특징]")
    print(f"- MTF 피보나치 컨플루언스 기반 지지/저항 감지")
    print(f"- 물리학 지표 (모멘텀, 에너지, 엔트로피) 검증")
    print(f"- RSI 다이버전스 + Stoch RSI 과매수/과매도 분석")
    print(f"- 와이코프 페이즈 판정 (매집/분산/추세)")
    print(f"- 선물 10x 레버리지 격리 마진")

    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 단일 심볼 테스트
        symbol = sys.argv[1]
        run_navigator_backtest(symbol)
    else:
        # 멀티 심볼 테스트
        run_multi_symbol_backtest()
