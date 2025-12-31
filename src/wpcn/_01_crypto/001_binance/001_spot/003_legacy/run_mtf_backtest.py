#!/usr/bin/env python
"""
MTF 차트 네비게이터 V3 백테스트
Multi-Timeframe Chart Navigator V3 Backtest

구조:
- 5분봉 데이터 기반
- 15분/1시간/4시간 리샘플링
- 1년 백테스트
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
from wpcn.navigation.chart_navigator_v3 import ChartNavigatorV3
from wpcn.execution.futures_broker import FuturesBroker


def run_single_backtest(symbol: str, days: int = 365):
    """단일 코인 MTF 백테스트"""
    print(f'\n{"="*70}')
    print(f'{symbol} MTF 네비게이터 V3 백테스트 ({days}일)')
    print(f'{"="*70}')

    safe_symbol = symbol.replace('/', '-')
    data_path = REPO_ROOT / 'data' / 'raw' / f'binance_{safe_symbol}_5m_{days}d.parquet'
    data_path.parent.mkdir(parents=True, exist_ok=True)

    # 5분봉 데이터 다운로드
    print(f'[1/3] 5분봉 데이터 다운로드 중...')
    try:
        fetch_ohlcv_to_parquet('binance', symbol, '5m', days, str(data_path))
        df = load_parquet(str(data_path))
        print(f'[완료] {len(df):,} 캔들 로드 (5분봉)')
    except Exception as e:
        print(f'[오류] 데이터 다운로드 실패: {e}')
        return None

    # 네비게이터 V3 초기화
    print(f'[2/3] MTF 네비게이터 V3 초기화...')
    navigator = ChartNavigatorV3(
        rsi_period=14,
        stoch_period=14,
        wyckoff_lookback=50,
        atr_period=14
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

    # 5분봉 기준 설정 (속도 최적화)
    warmup = 1000  # 4시간봉 계산을 위한 충분한 워밍업
    total_bars = len(df)
    trades_executed = 0
    long_trades = 0
    short_trades = 0
    last_trade_bar = -100
    cooldown_bars = 72  # 6시간 쿨다운 (5분봉 기준)
    analysis_interval = 24  # 2시간마다 분석 (5분봉 24개) - 속도 최적화

    signal_count = 0
    cached_signal = None
    last_analysis_idx = 0

    for i in range(warmup, total_bars):
        t = df.index[i]
        o, h, l, c = df.iloc[i][['open', 'high', 'low', 'close']]

        # 바 처리
        event = broker.process_bar(t, i, o, h, l, c, max_hold_bars=144)  # 12시간 최대
        if event:
            trades_executed += 1

        # 분석 (2시간마다)
        if i % analysis_interval == 0:
            # 최근 데이터만 사용 (메모리 절약 + 속도)
            lookback = min(i + 1, 5000)  # 최대 5000봉만 사용
            current_df = df.iloc[i+1-lookback:i+1]
            try:
                cached_signal = navigator.get_signal(current_df)
                last_analysis_idx = i
                if cached_signal:
                    signal_count += 1
            except Exception:
                cached_signal = None

        # 진입 로직
        if broker.position is None and (i - last_trade_bar) >= cooldown_bars:
            # 최근 분석 결과 사용 (30분 이내)
            if cached_signal and (i - last_analysis_idx) < analysis_interval * 2:
                if cached_signal.action == 'LONG':
                    success = broker.open_position(
                        t, i, c, 'long', 0.15,
                        cached_signal.stop_loss,
                        cached_signal.take_profit,
                        f"MTF_LONG_{cached_signal.timeframe_alignment}TF"
                    )
                    if success:
                        trades_executed += 1
                        long_trades += 1
                        last_trade_bar = i
                        cached_signal = None

                elif cached_signal.action == 'SHORT':
                    success = broker.open_position(
                        t, i, c, 'short', 0.15,
                        cached_signal.stop_loss,
                        cached_signal.take_profit,
                        f"MTF_SHORT_{cached_signal.timeframe_alignment}TF"
                    )
                    if success:
                        trades_executed += 1
                        short_trades += 1
                        last_trade_bar = i
                        cached_signal = None

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
    print(f'결과: {symbol} ({days}일)')
    print(f'{"─"*70}')
    print(f'  전략: MTF 차트 네비게이터 V3')
    print(f'  타임프레임: 5M(진입) / 15M(다이버전스) / 1H(와이코프) / 4H(추세)')
    print(f'  레버리지: 5x')
    print(f'  초기 자본: $10,000')
    print(f'  최종 자본: ${stats.get("final_balance", broker.balance):,.0f}')
    print(f'  총 수익률: {stats.get("total_return", 0):+.2f}%')
    print(f'  최대 낙폭: {stats.get("max_drawdown", 0):.2f}%')
    print(f'  총 거래: {stats.get("total_trades", 0)}회 (롱:{long_trades} / 숏:{short_trades})')
    print(f'  승률: {stats.get("win_rate", 0):.1f}%')
    print(f'  손익비: {stats.get("profit_factor", 0):.2f}')
    print(f'  청산: {stats.get("liquidations", 0)}회')
    print(f'  신호 발생: {signal_count}회')
    print(f'{"─"*70}')

    return {
        'symbol': symbol,
        'stats': stats,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'signal_count': signal_count
    }


def main():
    """BTC 1년 백테스트"""
    symbols = ["BTC/USDT"]  # 1개 코인만 먼저 테스트
    days = 365

    start_time = datetime.now()

    print('#' * 70)
    print('#  MTF 차트 네비게이터 V3 백테스트')
    print('#  5M(진입) / 15M(다이버전스) / 1H(와이코프) / 4H(추세)')
    print('#' * 70)
    print(f'시작 시간: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'심볼: {", ".join(symbols)}')
    print(f'기간: {days}일 (1년)')
    print(f'설정: 5x 레버리지 | MTF 정렬 필요')
    print('#' * 70)

    results = {}

    for symbol in symbols:
        result = run_single_backtest(symbol, days)
        if result:
            results[symbol] = result

    end_time = datetime.now()
    elapsed = end_time - start_time

    # 종합 결과
    if results:
        print('\n')
        print('=' * 70)
        print('                         종합 결과')
        print('=' * 70)
        print(f'{"심볼":<12} {"수익률":>10} {"MDD":>8} {"거래":>6} {"롱":>4} {"숏":>4} {"승률":>8} {"손익비":>8}')
        print('-' * 70)

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

        print('=' * 70)

    print(f'\n시작: {start_time.strftime("%H:%M:%S")}')
    print(f'종료: {end_time.strftime("%H:%M:%S")}')
    print(f'소요: {elapsed}')

    print('\n[MTF 전략 구조]')
    print('  4H: 대추세 필터 (EMA 50/200)')
    print('  1H: 와이코프 구조 (매집/분산/스프링/업쓰러스트)')
    print('  15M: RSI 다이버전스 + 피보나치')
    print('  5M: Stoch RSI 진입 타이밍 + 반전 캔들')

    return results


if __name__ == "__main__":
    main()
