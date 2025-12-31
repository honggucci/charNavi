#!/usr/bin/env python
"""
초고속 3코인 1년 백테스트 (약 1시간)
Fast 3-coin 1-year backtest
"""
import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from wpcn.core.types import BacktestCosts
from wpcn.data.ccxt_fetch import fetch_ohlcv_to_parquet
from wpcn.data.loaders import load_parquet
from wpcn.navigation.chart_navigator import ChartNavigator
from wpcn.execution.futures_broker import FuturesBroker


def run_single_backtest(symbol: str, days: int = 365):
    """단일 코인 초고속 백테스트"""
    print(f'\n{"="*70}')
    print(f'{symbol} 백테스트 ({days}일, 1년)')
    print(f'{"="*70}')

    safe_symbol = symbol.replace('/', '-')
    data_path = REPO_ROOT / 'data' / 'raw' / f'binance_{safe_symbol}_15m_{days}d.parquet'
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # 데이터 다운로드
    print(f'[1/3] 데이터 다운로드 중...')
    try:
        fetch_ohlcv_to_parquet('binance', symbol, '15m', days, str(data_path))
        df = load_parquet(str(data_path))
        print(f'[완료] {len(df):,} 캔들 로드')
    except Exception as e:
        print(f'[오류] 데이터 다운로드 실패: {e}')
        return None

    # 네비게이터 초기화 (초고속 설정)
    print(f'[2/3] 네비게이터 초기화...')
    navigator = ChartNavigator(
        timeframes=['15m', '1h', '4h'],  # 1d 제외
        confluence_tolerance=0.5,
        min_confluence=2
    )

    costs = BacktestCosts(fee_bps=4.0, slippage_bps=5.0)
    broker = FuturesBroker(
        initial_balance=10000.0,
        leverage=5.0,
        margin_mode='isolated',
        costs=costs
    )

    # 백테스트 실행
    print(f'[3/3] 백테스트 실행 중...')

    warmup = 200
    total_bars = len(df)
    trades_executed = 0
    long_trades = 0
    short_trades = 0
    last_trade_bar = -100
    cooldown_bars = 32  # 8시간 쿨다운
    analysis_interval = 96  # 24시간마다 분석 (초고속)

    cached_report = None

    for i in range(warmup, total_bars):
        t = df.index[i]
        o, h, l, c = df.iloc[i][['open', 'high', 'low', 'close']]

        # 바 처리
        event = broker.process_bar(t, i, o, h, l, c, max_hold_bars=64)
        if event:
            trades_executed += 1

        # 진입 로직
        if broker.position is None and (i - last_trade_bar) >= cooldown_bars:
            if i % analysis_interval == 0:
                current_df = df.iloc[:i+1]
                try:
                    cached_report = navigator.analyze(current_df)
                except:
                    cached_report = None

            if cached_report:
                action = cached_report.recommended_action
                confidence = cached_report.overall_confidence

                # LONG 신호
                if action == 'LONG_NOW' and confidence > 0.6:
                    entry = c
                    sl = entry * 0.98  # -2%
                    tp = entry * 1.04  # +4%

                    if broker.open_position(t, i, entry, 'long', 0.15, sl, tp, 'NAV_LONG'):
                        trades_executed += 1
                        long_trades += 1
                        last_trade_bar = i
                        cached_report = None

                # SHORT 신호
                elif action == 'SHORT_NOW' and confidence > 0.6:
                    entry = c
                    sl = entry * 1.02
                    tp = entry * 0.96

                    if broker.open_position(t, i, entry, 'short', 0.15, sl, tp, 'NAV_SHORT'):
                        trades_executed += 1
                        short_trades += 1
                        last_trade_bar = i
                        cached_report = None

        # 진행률 (10000봉마다)
        if i % 10000 == 0:
            progress = (i - warmup) / (total_bars - warmup) * 100
            equity = broker.get_equity(c)
            print(f'  {progress:5.1f}% | ${equity:>10,.0f} | L:{long_trades} S:{short_trades}')

    # 최종 청산
    if broker.position:
        broker.close_position(df.index[-1], df.iloc[-1]['close'], 'TIME', 'FINAL')

    stats = broker.get_statistics()

    # 결과 출력
    print(f'\n{"─"*70}')
    print(f'결과: {symbol} (1년)')
    print(f'{"─"*70}')
    print(f'  레버리지: 5x')
    print(f'  TP/SL: 4%/2% (손익비 2:1)')
    print(f'  초기 자본: $10,000')
    print(f'  최종 자본: ${stats.get("final_balance", broker.balance):,.0f}')
    print(f'  총 수익률: {stats.get("total_return", 0):+.2f}%')
    print(f'  최대 낙폭: {stats.get("max_drawdown", 0):.2f}%')
    print(f'  총 거래: {stats.get("total_trades", 0)}회 (롱:{long_trades} / 숏:{short_trades})')
    print(f'  승률: {stats.get("win_rate", 0):.1f}%')
    print(f'  손익비: {stats.get("profit_factor", 0):.2f}')
    print(f'  청산: {stats.get("liquidations", 0)}회')
    print(f'{"─"*70}')

    return {
        'symbol': symbol,
        'stats': stats,
        'long_trades': long_trades,
        'short_trades': short_trades
    }


def main():
    """3개 코인 1년 백테스트 실행"""
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    days = 365

    start_time = datetime.now()

    print('#' * 70)
    print('#  확률론적 와이코프 스나이퍼 - 차트 네비게이터')
    print('#  3코인 1년 선물 거래 백테스트 (초고속 모드)')
    print('#' * 70)
    print(f'시작 시간: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'심볼: {", ".join(symbols)}')
    print(f'기간: {days}일 (1년)')
    print(f'설정: 5x 레버리지 | TP 4% | SL 2% | 분석주기 24시간')
    print('#' * 70)

    results = {}

    for symbol in symbols:
        result = run_single_backtest(symbol, days)
        if result:
            results[symbol] = result

    end_time = datetime.now()
    elapsed = end_time - start_time

    # 종합 결과
    print('\n')
    print('=' * 70)
    print('                         종합 결과')
    print('=' * 70)
    print(f'{"심볼":<12} {"수익률":>10} {"MDD":>8} {"거래":>6} {"롱":>4} {"숏":>4} {"승률":>8} {"손익비":>8}')
    print('-' * 70)

    total_return = 0
    total_trades = 0
    total_long = 0
    total_short = 0

    for symbol, result in results.items():
        stats = result['stats']
        ret = stats.get('total_return', 0)
        mdd = stats.get('max_drawdown', 0)
        trades = stats.get('total_trades', 0)
        long_t = result['long_trades']
        short_t = result['short_trades']
        winrate = stats.get('win_rate', 0)
        pf = stats.get('profit_factor', 0)

        print(f'{symbol:<12} {ret:>+9.2f}% {mdd:>7.2f}% {trades:>6d} {long_t:>4d} {short_t:>4d} {winrate:>7.1f}% {pf:>8.2f}')

        total_return += ret
        total_trades += trades
        total_long += long_t
        total_short += short_t

    print('-' * 70)
    avg_return = total_return / len(results) if results else 0
    print(f'{"합계":<12} {total_return:>+9.2f}%')
    print(f'{"평균":<12} {avg_return:>+9.2f}%')
    print(f'{"총 거래":<12} {total_trades:>6d} {total_long:>4d} {total_short:>4d}')
    print('=' * 70)

    print(f'\n시작: {start_time.strftime("%H:%M:%S")}')
    print(f'종료: {end_time.strftime("%H:%M:%S")}')
    print(f'소요: {elapsed}')

    # 분석
    print('\n[분석]')
    if avg_return > 20:
        print('★ 평균 수익률이 매우 우수합니다!')
    elif avg_return > 10:
        print('◎ 평균 수익률이 우수합니다.')
    elif avg_return > 0:
        print('○ 수익을 내고 있습니다.')
    else:
        print('△ 전략 개선이 필요합니다.')

    return results


if __name__ == "__main__":
    main()
