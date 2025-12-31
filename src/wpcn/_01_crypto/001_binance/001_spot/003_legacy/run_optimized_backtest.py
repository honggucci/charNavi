#!/usr/bin/env python
"""
최적화된 5코인 백테스트
Optimized 5-coin backtest with improved parameters
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
    """단일 코인 최적화 백테스트"""
    print(f'\n{"="*60}')
    print(f'{symbol} 백테스트 시작 ({days}일)')
    print(f'{"="*60}')

    safe_symbol = symbol.replace('/', '-')
    data_path = REPO_ROOT / 'data' / 'raw' / f'binance_{safe_symbol}_15m_{days}d.parquet'
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # 데이터 다운로드
    try:
        fetch_ohlcv_to_parquet('binance', symbol, '15m', days, str(data_path))
        df = load_parquet(str(data_path))
        print(f'[데이터] {len(df)} 캔들 로드')
    except Exception as e:
        print(f'[오류] 데이터 다운로드 실패: {e}')
        return None

    # 네비게이터 초기화 (최적화된 설정)
    navigator = ChartNavigator(
        timeframes=['15m', '1h', '4h'],  # 1d 제외 (속도 개선)
        confluence_tolerance=0.5,
        min_confluence=2
    )

    costs = BacktestCosts(fee_bps=4.0, slippage_bps=5.0)
    broker = FuturesBroker(
        initial_balance=10000.0,
        leverage=5.0,  # 보수적 레버리지
        margin_mode='isolated',
        costs=costs
    )

    warmup = 200
    total_bars = len(df)
    trades_executed = 0
    last_trade_bar = -50
    cooldown_bars = 24  # 6시간 쿨다운
    analysis_interval = 32  # 8시간마다 분석 (속도 최적화)

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
                        last_trade_bar = i
                        cached_report = None

                # SHORT 신호
                elif action == 'SHORT_NOW' and confidence > 0.6:
                    entry = c
                    sl = entry * 1.02
                    tp = entry * 0.96

                    if broker.open_position(t, i, entry, 'short', 0.15, sl, tp, 'NAV_SHORT'):
                        trades_executed += 1
                        last_trade_bar = i
                        cached_report = None

        # 진행률
        if i % 5000 == 0:
            progress = (i - warmup) / (total_bars - warmup) * 100
            equity = broker.get_equity(c)
            print(f'  {symbol}: {progress:.0f}% | ${equity:,.0f} | 거래:{trades_executed}')

    # 최종 청산
    if broker.position:
        broker.close_position(df.index[-1], df.iloc[-1]['close'], 'TIME', 'FINAL')

    stats = broker.get_statistics()

    # 결과 출력
    print(f'\n[결과] {symbol}')
    print(f'  수익률: {stats.get("total_return", 0):+.2f}%')
    print(f'  MDD: {stats.get("max_drawdown", 0):.2f}%')
    print(f'  거래: {stats.get("total_trades", 0)}회')
    print(f'  승률: {stats.get("win_rate", 0):.1f}%')
    print(f'  손익비: {stats.get("profit_factor", 0):.2f}')

    return stats


def main():
    """5개 코인 백테스트 실행"""
    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "TRX/USDT", "BNB/USDT"]
    days = 180  # 6개월 (속도 최적화)

    print('#' * 70)
    print('# 확률론적 와이코프 스나이퍼 - 차트 네비게이터')
    print('# 5코인 6개월 선물 거래 백테스트')
    print('#' * 70)
    print(f'시작 시간: {datetime.now()}')
    print(f'심볼: {", ".join(symbols)}')
    print(f'설정: 5x 레버리지, TP 4%, SL 2%')
    print('#' * 70)

    results = {}

    for symbol in symbols:
        stats = run_single_backtest(symbol, days)
        if stats:
            results[symbol] = stats

    # 종합 결과
    print('\n' + '=' * 80)
    print('종합 결과')
    print('=' * 80)
    print(f'{"심볼":<12} {"수익률":>12} {"MDD":>10} {"거래":>8} {"승률":>10} {"손익비":>10}')
    print('-' * 80)

    total_return = 0
    total_trades = 0

    for symbol, stats in results.items():
        ret = stats.get('total_return', 0)
        mdd = stats.get('max_drawdown', 0)
        trades = stats.get('total_trades', 0)
        winrate = stats.get('win_rate', 0)
        pf = stats.get('profit_factor', 0)

        print(f'{symbol:<12} {ret:>+11.2f}% {mdd:>9.2f}% {trades:>8d} {winrate:>9.1f}% {pf:>10.2f}')

        total_return += ret
        total_trades += trades

    print('-' * 80)
    avg_return = total_return / len(results) if results else 0
    print(f'{"합계":<12} {total_return:>+11.2f}%')
    print(f'{"평균":<12} {avg_return:>+11.2f}%')
    print(f'{"총 거래":<12} {total_trades:>8d}')
    print('=' * 80)

    print(f'\n종료 시간: {datetime.now()}')

    # 분석
    if avg_return > 10:
        print('[+] 평균 수익률이 우수합니다!')
    elif avg_return > 0:
        print('[=] 수익을 내고 있습니다.')
    else:
        print('[-] 전략 개선이 필요합니다.')

    return results


if __name__ == "__main__":
    main()
