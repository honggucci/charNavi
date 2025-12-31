#!/usr/bin/env python
"""빠른 단일 코인 테스트"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from wpcn.core.types import BacktestCosts
from wpcn.data.loaders import load_parquet
from wpcn.navigation.chart_navigator import ChartNavigator
from wpcn.execution.futures_broker import FuturesBroker

# 데이터 로드
df = load_parquet(str(REPO_ROOT / 'data' / 'raw' / 'binance_BTC-USDT_15m_90d.parquet'))
print(f'[데이터] {len(df)} 캔들')

# 개선된 파라미터
navigator = ChartNavigator(timeframes=['15m', '1h', '4h'], confluence_tolerance=0.5, min_confluence=2)
costs = BacktestCosts(fee_bps=4.0, slippage_bps=5.0)
broker = FuturesBroker(initial_balance=10000.0, leverage=5.0, margin_mode='isolated', costs=costs)

warmup = 200
total_bars = len(df)
trades_executed = 0
last_trade_bar = -50
cooldown_bars = 24
analysis_interval = 16

cached_report = None

for i in range(warmup, total_bars):
    t = df.index[i]
    o, h, l, c = df.iloc[i][['open', 'high', 'low', 'close']]

    event = broker.process_bar(t, i, o, h, l, c, max_hold_bars=64)
    if event:
        trades_executed += 1

    if broker.position is None and (i - last_trade_bar) >= cooldown_bars:
        if i % analysis_interval == 0:
            current_df = df.iloc[:i+1]
            cached_report = navigator.analyze(current_df)

        if cached_report:
            action = cached_report.recommended_action
            confidence = cached_report.overall_confidence

            if action == 'LONG_NOW' and confidence > 0.6:
                entry = c
                sl = entry * 0.98
                tp = entry * 1.04

                if broker.open_position(t, i, entry, 'long', 0.15, sl, tp, 'NAV_LONG'):
                    trades_executed += 1
                    last_trade_bar = i
                    cached_report = None

            elif action == 'SHORT_NOW' and confidence > 0.6:
                entry = c
                sl = entry * 1.02
                tp = entry * 0.96

                if broker.open_position(t, i, entry, 'short', 0.15, sl, tp, 'NAV_SHORT'):
                    trades_executed += 1
                    last_trade_bar = i
                    cached_report = None

    if i % 2000 == 0:
        print(f'  진행: {(i-warmup)/(total_bars-warmup)*100:.0f}% | 거래: {trades_executed}')

if broker.position:
    broker.close_position(df.index[-1], df.iloc[-1]['close'], 'TIME', 'FINAL')

stats = broker.get_statistics()
print()
print('='*60)
print('개선된 전략 결과: BTC/USDT (90일)')
print('='*60)
print(f'레버리지: 5x (보수적)')
print(f'TP/SL: 4%/2% (2:1 비율)')
print(f'초기 자본: $10,000')
print(f'최종 자본: ${stats.get("final_balance", broker.balance):,.0f}')
print(f'총 수익률: {stats.get("total_return", 0):+.2f}%')
print(f'최대 낙폭: {stats.get("max_drawdown", 0):.2f}%')
print(f'총 거래: {stats.get("total_trades", 0)}회')
print(f'승률: {stats.get("win_rate", 0):.1f}%')
print(f'손익비: {stats.get("profit_factor", 0):.2f}')
print(f'청산: {stats.get("liquidations", 0)}회')
print('='*60)
