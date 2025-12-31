#!/usr/bin/env python
"""
차트 네비게이터 V2 백테스트
Chart Navigator V2 Backtest

핵심 구조:
1. 와이코프 패턴 = 구조의 뼈대
2. Stoch RSI = 다이버전스 필터
3. RSI 다이버전스 = 스나이핑 타이밍
4. 지그재그 = 확률적 선행지표
5. MTF 추세 확인 = 안전장치
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
from wpcn.navigation.chart_navigator_v2 import ChartNavigatorV2
from wpcn.execution.futures_broker import FuturesBroker


def run_single_backtest(symbol: str, days: int = 90):
    """단일 코인 네비게이터 V2 백테스트"""
    print(f'\n{"="*70}')
    print(f'{symbol} 차트 네비게이터 V2 백테스트 ({days}일)')
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

    # 네비게이터 V2 초기화
    print(f'[2/3] 네비게이터 V2 초기화...')
    navigator = ChartNavigatorV2(
        rsi_period=14,
        stoch_rsi_period=14,
        wyckoff_lookback=50,
        zigzag_threshold=0.02,
        atr_period=14,
        atr_sl_mult=1.5,
        atr_tp_mult=3.0
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
    last_trade_bar = -50
    cooldown_bars = 48  # 12시간 쿨다운 (과매매 방지)
    analysis_interval = 16  # 4시간마다 분석

    signal_count = 0

    for i in range(warmup, total_bars):
        t = df.index[i]
        o, h, l, c = df.iloc[i][['open', 'high', 'low', 'close']]

        # 바 처리
        event = broker.process_bar(t, i, o, h, l, c, max_hold_bars=64)  # 16시간 최대
        if event:
            trades_executed += 1

        # 진입 로직
        if broker.position is None and (i - last_trade_bar) >= cooldown_bars:
            if i % analysis_interval == 0:
                current_df = df.iloc[:i+1]
                try:
                    signal = navigator.get_signal(current_df)

                    if signal:
                        signal_count += 1

                        if signal.action == 'LONG':
                            success = broker.open_position(
                                t, i, c, 'long', signal.position_size_pct,
                                signal.stop_loss, signal.take_profit,
                                f"NAV2_LONG"
                            )
                            if success:
                                trades_executed += 1
                                long_trades += 1
                                last_trade_bar = i

                        elif signal.action == 'SHORT':
                            success = broker.open_position(
                                t, i, c, 'short', signal.position_size_pct,
                                signal.stop_loss, signal.take_profit,
                                f"NAV2_SHORT"
                            )
                            if success:
                                trades_executed += 1
                                short_trades += 1
                                last_trade_bar = i

                except Exception:
                    pass

        # 진행률 (2000봉마다)
        if i % 2000 == 0:
            progress = (i - warmup) / (total_bars - warmup) * 100
            equity = broker.get_equity(c)
            print(f'  {progress:5.1f}% | ${equity:>10,.0f} | L:{long_trades} S:{short_trades}')

    # 최종 청산
    if broker.position:
        broker.close_position(df.index[-1], df.iloc[-1]['close'], 'TIME', 'FINAL')

    stats = broker.get_statistics()

    # 결과 출력
    print(f'\n{"─"*70}')
    print(f'결과: {symbol} ({days}일)')
    print(f'{"─"*70}')
    print(f'  전략: 차트 네비게이터 V2')
    print(f'  - 와이코프 구조 + Stoch RSI 필터 다이버전스')
    print(f'  - 지그재그 확률 예측 + ATR 동적 TP/SL')
    print(f'  레버리지: 5x')
    print(f'  초기 자본: $10,000')
    print(f'  최종 자본: ${stats.get("final_balance", broker.balance):,.0f}')
    print(f'  총 수익률: {stats.get("total_return", 0):+.2f}%')
    print(f'  최대 낙폭: {stats.get("max_drawdown", 0):.2f}%')
    print(f'  총 거래: {stats.get("total_trades", 0)}회 (롱:{long_trades} / 숏:{short_trades})')
    print(f'  승률: {stats.get("win_rate", 0):.1f}%')
    print(f'  손익비: {stats.get("profit_factor", 0):.2f}')
    print(f'  청산: {stats.get("liquidations", 0)}회')
    print(f'  신호: {signal_count}회')
    print(f'{"─"*70}')

    return {
        'symbol': symbol,
        'stats': stats,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'signal_count': signal_count
    }


def main():
    """3개 코인 백테스트 실행"""
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    days = 90

    start_time = datetime.now()

    print('#' * 70)
    print('#  차트 네비게이터 V2 백테스트')
    print('#  와이코프 + Stoch RSI 필터 다이버전스 + 지그재그 예측')
    print('#' * 70)
    print(f'시작 시간: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'심볼: {", ".join(symbols)}')
    print(f'기간: {days}일')
    print(f'설정: 5x 레버리지 | ATR 기반 동적 TP/SL')
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
    print('\n[전략 특징]')
    print('  1. 와이코프 구조: 매집/분산/스프링/업쓰러스트 감지')
    print('  2. Stoch RSI 필터: 과매수/과매도 구간에서만 다이버전스 체크')
    print('  3. 지그재그 예측: 와이코프+다이버전스 기반 확률적 예측')
    print('  4. ATR 동적 TP/SL: 변동성에 따른 적응형 손익비')
    print('  5. 상위 TF 추세: 역추세 매매 페널티')

    if avg_return > 10:
        print('\n★ 평균 수익률이 우수합니다!')
    elif avg_return > 0:
        print('\n○ 수익을 내고 있습니다.')
    else:
        print('\n△ 전략 개선이 필요합니다.')

    return results


if __name__ == "__main__":
    main()
