#!/usr/bin/env python
"""
간편한 백테스팅 메인 스크립트 - 한 번에 결과까지 보기
"""
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from wpcn.core.types import (
    BacktestCosts, BacktestConfig, RunConfig,
    COSTS_SPOT_BTC, COSTS_SPOT_ALT, CONFIG_NAVIGATION, CONFIG_SCALPING
)
from wpcn.data.ccxt_fetch import fetch_ohlcv_to_parquet
from wpcn.data.loaders import load_parquet
from wpcn.engine.backtest import BacktestEngine, DEFAULT_THETA
from wpcn.reporting.charts import plot_backtest_results

def main():
    """일봉(1d) 기본값으로 백테스팅 실행 및 결과 표시"""
    
    # 설정값
    exchange = "binance"  # 현물 거래소
    symbol = "BTC/USDT"  # 비트코인
    timeframe = "15m"  # 15분봉 (개선된 단타용)
    days = 30  # 30일 데이터
    
    print(f"\n{'='*60}")
    print(f"WPCN 백테스팅 시작")
    print(f"{'='*60}")
    print(f"거래소: {exchange}")
    print(f"심볼: {symbol}")
    print(f"타임프레임: {timeframe}")
    print(f"기간: {days}일")
    print(f"{'='*60}\n")
    
    # 데이터 경로
    safe_symbol = symbol.replace("/", "-")
    data_path = REPO_ROOT / "data" / "raw" / f"{exchange}_{safe_symbol}_{timeframe}.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 데이터 다운로드
    print(f"[1/4] 데이터 다운로드 중...")
    if not data_path.exists():
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
    else:
        print(f"[완료] 기존 데이터 사용: {data_path}")
    
    # 데이터 로드
    print(f"\n[2/4] 데이터 로드 중...")
    df = load_parquet(str(data_path))
    print(f"[완료] 데이터 로드 완료: {len(df)} 캔들")
    
    # 백테스팅 설정
    print(f"\n[3/4] 백테스팅 실행 중...")

    # ============================================================================
    # 비용 설정 (현실화)
    # - 기존: fee=10bps + slip=30bps = 40bps (0.4%) → 너무 보수적
    # - 변경: BTC는 COSTS_SPOT_BTC (10.5bps), ALT는 COSTS_SPOT_ALT (20bps)
    # ============================================================================
    is_btc = "BTC" in symbol.upper()
    costs = COSTS_SPOT_BTC if is_btc else COSTS_SPOT_ALT
    print(f"비용 설정: fee={costs.fee_bps}bps, slip={costs.slippage_bps}bps (편도 {costs.total_one_way_pct():.3f}%)")

    # ============================================================================
    # Gate 설정 (상향 조정)
    # - 기존: conf=0.50, edge=0.60 → 쓰레기 신호 허용
    # - 변경: conf=0.65, edge=0.70 (Navigation), conf=0.60, edge=0.65 (Scalping)
    # 철학: "안 들어가는 날이 많아져야 계좌가 산다"
    # ============================================================================
    scalping_mode = True  # True: 단타 모드, False: 네비게이션 모드

    if scalping_mode:
        bt = BacktestConfig(
            initial_equity=1.0,
            max_hold_bars=30,           # 15분봉: 30봉 = 약 7.5시간
            tp1_frac=0.5,
            use_tp2=True,
            conf_min=0.60,              # 0.50 → 0.60 (상향)
            edge_min=0.65,              # 0.60 → 0.65 (상향)
            confirm_bars=1,             # 단타: 빠른 진입
            reclaim_hold_bars=1,        # 단타: 짧은 유지
            chaos_ret_atr=3.0,
            adx_len=14,
            adx_trend=15.0,
        )
        print(f"모드: Scalping (conf≥{bt.conf_min}, edge≥{bt.edge_min}, hold≤{bt.max_hold_bars}봉)")
    else:
        bt = BacktestConfig(
            initial_equity=1.0,
            max_hold_bars=192,          # 15분봉: 192봉 = 48시간
            tp1_frac=0.5,
            use_tp2=True,
            conf_min=0.65,              # 0.50 → 0.65 (상향)
            edge_min=0.70,              # 0.60 → 0.70 (상향)
            confirm_bars=2,             # 1 → 2 (휩쏘 방지)
            reclaim_hold_bars=2,        # NEW: reclaim 후 유지 확인
            chaos_ret_atr=3.0,
            adx_len=14,
            adx_trend=20.0,
        )
        print(f"모드: Navigation (conf≥{bt.conf_min}, edge≥{bt.edge_min}, hold≤{bt.max_hold_bars}봉)")

    # 다중 타임프레임 설정: 15분봉 기준으로 1시간봉, 4시간봉 확인
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
        mtf=mtf_timeframes,  # MTF 활성화
        data_path=str(data_path),
    )

    print(f"MTF 활성화: {', '.join(mtf_timeframes)} (상위 타임프레임 추세 확인)")
    print(f"단타 모드: {'활성화' if scalping_mode else '비활성화'}")

    # Phase A/B 양방향 축적 전략 활성화
    use_phase_accumulation = True  # 박스 피보나치 그리드 거미줄 + 양방향 안전마진
    print(f"Phase A/B 양방향 축적: {'활성화' if use_phase_accumulation else '비활성화'}")
    if use_phase_accumulation:
        print(f"  - 박스 피보나치 그리드 거미줄 (롱+숏)")
        print(f"  - 롱: fib_236, fib_382, fib_500 레벨에서 박스 하단 근처 매수")
        print(f"  - 숏: fib_618, fib_786 레벨에서 박스 상단 근처 매도")
        print(f"  - 방향별 비율: Accumulation(롱 70%/숏 30%), Distribution(롱 30%/숏 70%)")

    # 백테스트 실행
    engine = BacktestEngine(runs_root=str(REPO_ROOT / "runs"))
    run_id = engine.run(df, cfg, theta=DEFAULT_THETA, scalping_mode=scalping_mode, use_phase_accumulation=use_phase_accumulation)
    print(f"[완료] 백테스팅 완료: {run_id}")
    
    # 결과 경로
    run_dir = REPO_ROOT / "runs" / run_id
    print(f"\n[4/4] 결과 생성 중...")
    
    # 결과 데이터 로드
    try:
        equity_df = pd.read_parquet(run_dir / "equity.parquet")
        trades_df = pd.read_parquet(run_dir / "trades.parquet")
        nav_df = pd.read_parquet(run_dir / "nav.parquet")

        # 성과 지표 출력
        print(f"\n{'='*60}")
        print(f"성과 지표")
        print(f"{'='*60}")

        initial_equity = float(cfg.bt.initial_equity)
        final_equity = equity_df['equity'].iloc[-1]
        total_return = ((final_equity - initial_equity) / initial_equity) * 100

        print(f"초기 자본: ${initial_equity:.2f}")
        print(f"최종 자본: ${final_equity:.2f}")
        print(f"총 수익률: {total_return:.2f}%")

        # 최대 낙폭 계산
        equity_series = equity_df['equity']
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()
        print(f"최대 낙폭(MDD): {max_drawdown:.2f}%")

        # 거래 통계
        entry_trades = trades_df[trades_df['type'] == 'ENTRY']
        tp_trades = trades_df[trades_df['type'].isin(['TP1', 'TP2'])]
        stop_trades = trades_df[trades_df['type'] == 'STOP']
        time_exit_trades = trades_df[trades_df['type'] == 'TIME_EXIT']

        # 양방향 축적 거래 통계
        accum_long_trades = trades_df[trades_df['type'] == 'ACCUM_LONG']
        accum_short_trades = trades_df[trades_df['type'] == 'ACCUM_SHORT']
        accum_long_tp = trades_df[trades_df['type'] == 'ACCUM_LONG_TP']
        accum_long_stop = trades_df[trades_df['type'] == 'ACCUM_LONG_STOP']
        accum_short_tp = trades_df[trades_df['type'] == 'ACCUM_SHORT_TP']
        accum_short_stop = trades_df[trades_df['type'] == 'ACCUM_SHORT_STOP']

        # 레거시 호환
        accum_trades = trades_df[trades_df['type'].isin(['ACCUM', 'ACCUM_LONG', 'ACCUM_SHORT'])]
        accum_tp_trades = trades_df[trades_df['type'].isin(['ACCUM_TP', 'ACCUM_LONG_TP', 'ACCUM_SHORT_TP'])]
        accum_stop_trades = trades_df[trades_df['type'].isin(['ACCUM_STOP', 'ACCUM_LONG_STOP', 'ACCUM_SHORT_STOP'])]

        print(f"\n=== 메인 전략 (Phase C) ===")
        print(f"총 진입 횟수: {len(entry_trades)}")
        print(f"목표가 달성(TP): {len(tp_trades)}회")
        print(f"손절(STOP): {len(stop_trades)}회")
        print(f"시간 청산: {len(time_exit_trades)}회")

        if len(accum_trades) > 0:
            print(f"\n=== 양방향 축적 전략 (Phase A/B) ===")

            # 롱 축적 통계
            print(f"\n[롱 축적]")
            print(f"  롱 진입: {len(accum_long_trades)}회")
            if 'event' in accum_long_trades.columns and len(accum_long_trades) > 0:
                spider_long = len(accum_long_trades[accum_long_trades['event'].str.contains('spider', case=False, na=False)])
                safety_long = len(accum_long_trades[accum_long_trades['event'].str.contains('safety', case=False, na=False)])
                print(f"    - 피보나치 그리드: {spider_long}회")
                print(f"    - 안전마진: {safety_long}회")
            print(f"  롱 익절: {len(accum_long_tp)}회")
            print(f"  롱 손절: {len(accum_long_stop)}회")
            if len(accum_long_tp) + len(accum_long_stop) > 0:
                long_winrate = len(accum_long_tp) / (len(accum_long_tp) + len(accum_long_stop)) * 100
                print(f"  롱 승률: {long_winrate:.1f}%")

            # 숏 축적 통계
            print(f"\n[숏 축적]")
            print(f"  숏 진입: {len(accum_short_trades)}회")
            if 'event' in accum_short_trades.columns and len(accum_short_trades) > 0:
                spider_short = len(accum_short_trades[accum_short_trades['event'].str.contains('spider', case=False, na=False)])
                safety_short = len(accum_short_trades[accum_short_trades['event'].str.contains('safety', case=False, na=False)])
                print(f"    - 피보나치 그리드: {spider_short}회")
                print(f"    - 안전마진: {safety_short}회")
            print(f"  숏 익절: {len(accum_short_tp)}회")
            print(f"  숏 손절: {len(accum_short_stop)}회")
            if len(accum_short_tp) + len(accum_short_stop) > 0:
                short_winrate = len(accum_short_tp) / (len(accum_short_tp) + len(accum_short_stop)) * 100
                print(f"  숏 승률: {short_winrate:.1f}%")

            # 전체 축적 손익
            print(f"\n[축적 전체]")
            total_accum = len(accum_trades)
            print(f"  총 축적: {total_accum}회 (롱 {len(accum_long_trades)}, 숏 {len(accum_short_trades)})")
            if len(accum_tp_trades) + len(accum_stop_trades) > 0:
                accum_win = len(accum_tp_trades)
                accum_loss = len(accum_stop_trades)
                accum_winrate = accum_win / (accum_win + accum_loss) * 100
                print(f"  전체 승률: {accum_winrate:.1f}% ({accum_win}승/{accum_loss}패)")

        # 거래별 손익률 계산
        print(f"\n{'='*60}")
        print(f"거래 내역 및 손익 분석")
        print(f"{'='*60}")

        # 메인 거래만 필터링 (ACCUM 관련 제외)
        accum_types = ['ACCUM', 'ACCUM_LONG', 'ACCUM_SHORT', 'ACCUM_TP', 'ACCUM_STOP',
                       'ACCUM_LONG_TP', 'ACCUM_LONG_STOP', 'ACCUM_SHORT_TP', 'ACCUM_SHORT_STOP']
        main_trades = trades_df[~trades_df['type'].isin(accum_types)]

        if len(entry_trades) > 0:
            trade_groups = []
            current_entry = None
            current_exits = []

            for idx, trade in main_trades.iterrows():
                if trade['type'] == 'ENTRY':
                    if current_entry is not None:
                        trade_groups.append((current_entry, current_exits))
                    current_entry = trade
                    current_exits = []
                else:
                    current_exits.append(trade)

            if current_entry is not None:
                trade_groups.append((current_entry, current_exits))

            winning_trades = 0
            losing_trades = 0
            total_profit = 0
            total_loss = 0

            for trade_num, (entry, exits) in enumerate(trade_groups, 1):
                entry_price = entry['price']
                entry_time = entry['time']
                side = entry['side']
                event = entry.get('event', 'N/A')

                print(f"\n{'─'*60}")
                print(f"거래 #{trade_num}")
                print(f"{'─'*60}")
                print(f"[진입] {entry_time}")
                print(f"  방향: {'롱(매수)' if side == 'long' else '숏(매도)'}")
                print(f"  가격: ${entry_price:,.2f}")
                print(f"  이벤트: {event}")

                trade_pnl = 0
                for exit_trade in exits:
                    exit_price = exit_trade['price']
                    exit_time = exit_trade['time']
                    exit_type = exit_trade['type']

                    if side == 'long':
                        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:  # short
                        pnl_pct = ((entry_price - exit_price) / entry_price) * 100

                    # TP1은 50%, TP2는 나머지 50%
                    if exit_type == 'TP1':
                        weight = 0.5
                    else:
                        weight = 0.5 if 'TP1' in [e['type'] for e in exits] else 1.0

                    trade_pnl += pnl_pct * weight

                    status = "[+]" if pnl_pct > 0 else "[-]"
                    print(f"\n  [{exit_type}] {exit_time}")
                    print(f"    청산가격: ${exit_price:,.2f}")
                    print(f"    손익률: {status} {pnl_pct:+.2f}% (가중치: {weight*100:.0f}%)")

                print(f"\n  [거래 총 손익률] {trade_pnl:+.2f}%")

                if trade_pnl > 0:
                    winning_trades += 1
                    total_profit += trade_pnl
                    print(f"  결과: 수익 [WIN]")
                else:
                    losing_trades += 1
                    total_loss += abs(trade_pnl)
                    print(f"  결과: 손실 [LOSS]")

            # 통계 요약
            print(f"\n{'='*60}")
            print(f"거래 통계 요약")
            print(f"{'='*60}")
            total_trades = winning_trades + losing_trades
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            avg_profit = (total_profit / winning_trades) if winning_trades > 0 else 0
            avg_loss = (total_loss / losing_trades) if losing_trades > 0 else 0
            profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')

            print(f"총 거래 횟수: {total_trades}회")
            print(f"승리 거래: {winning_trades}회")
            print(f"패배 거래: {losing_trades}회")
            print(f"승률: {win_rate:.2f}%")
            print(f"평균 수익: +{avg_profit:.2f}%")
            print(f"평균 손실: -{avg_loss:.2f}%")
            print(f"손익비 (Profit Factor): {profit_factor:.2f}")

            # 시사점
            print(f"\n{'='*60}")
            print(f"분석 및 시사점")
            print(f"{'='*60}")

            if win_rate >= 60:
                print(f"[+] 승률이 {win_rate:.1f}%로 우수합니다.")
            elif win_rate >= 40:
                print(f"[=] 승률이 {win_rate:.1f}%로 보통 수준입니다.")
            else:
                print(f"[-] 승률이 {win_rate:.1f}%로 개선이 필요합니다.")

            if profit_factor >= 2.0:
                print(f"[+] 손익비가 {profit_factor:.2f}로 매우 우수합니다.")
            elif profit_factor >= 1.5:
                print(f"[+] 손익비가 {profit_factor:.2f}로 양호합니다.")
            elif profit_factor >= 1.0:
                print(f"[=] 손익비가 {profit_factor:.2f}로 보통입니다.")
            else:
                print(f"[-] 손익비가 {profit_factor:.2f}로 개선이 필요합니다.")

            if max_drawdown > -10:
                print(f"[+] 최대 낙폭 {max_drawdown:.2f}%로 리스크 관리가 양호합니다.")
            elif max_drawdown > -20:
                print(f"[=] 최대 낙폭 {max_drawdown:.2f}%로 적절한 수준입니다.")
            else:
                print(f"[-] 최대 낙폭 {max_drawdown:.2f}%로 리스크가 큽니다.")

            print(f"\n[전략 특징]")
            print(f"- Wyckoff Spring/UTAD 이벤트 기반 진입")
            print(f"- 분할 익절 전략 (TP1: 50%, TP2: 50%)")
            print(f"- 최대 보유 기간: {cfg.bt.max_hold_bars}봉")

            print(f"\n[개선 제안]")
            if win_rate < 50:
                print(f"- 진입 조건 강화 (conf_min, edge_min 조정)")
            if avg_loss > avg_profit * 2:
                print(f"- 손절 기준 최적화 필요")
            if total_trades < 10:
                print(f"- 거래 빈도가 낮음, 파라미터 완화 고려")
            if total_trades > 100:
                print(f"- 거래 빈도가 높음, 진입 조건 강화 고려")

        else:
            print("거래 내역이 없습니다.")

        print(f"\n{'='*60}")

        # 차트 생성
        plot_backtest_results(run_dir, df, equity_df, trades_df, nav_df)
        print(f"[완료] 차트 생성 완료")
    except Exception as e:
        print(f"[경고] 차트 생성 중 오류: {e}")
    
    # 결과 요약
    print(f"\n{'='*60}")
    print(f"백테스팅 완료!")
    print(f"{'='*60}")
    print(f"결과 폴더: {run_dir}")
    print(f"차트: {run_dir / 'charts'}")
    print(f"리포트: {run_dir / 'reports'}")
    print(f"{'='*60}\n")

def run_multi_symbol_test():
    """BTC, ETH, XRP, TRX, BNB에 대한 멀티 심볼 백테스팅"""

    symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "TRX/USDT", "BNB/USDT"]
    exchange = "binance"
    timeframe = "15m"
    days = 90  # 3개월

    results = {}

    print(f"\n{'='*70}")
    print(f"멀티 심볼 백테스팅: Wyckoff 양방향 축적 전략")
    print(f"{'='*70}")
    print(f"심볼: {', '.join(symbols)}")
    print(f"타임프레임: {timeframe}")
    print(f"기간: {days}일")
    print(f"슬리피지: 0.3% (30bps)")
    print(f"{'='*70}\n")

    for symbol in symbols:
        print(f"\n{'#'*70}")
        print(f"# {symbol} 백테스팅")
        print(f"{'#'*70}")

        # 데이터 경로
        safe_symbol = symbol.replace("/", "-")
        data_path = REPO_ROOT / "data" / "raw" / f"{exchange}_{safe_symbol}_{timeframe}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)

        # 데이터 다운로드 (강제 재다운로드)
        fetch_ohlcv_to_parquet(
            exchange_id=exchange,
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            out_path=str(data_path),
            api_key=None,
            secret=None,
        )

        # 데이터 로드
        df = load_parquet(str(data_path))

        # 백테스팅 설정 - 슬리피지 0.3% (30bps)
        costs = BacktestCosts(fee_bps=10.0, slippage_bps=30.0)  # 수수료 0.1% + 슬리피지 0.3%
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

        cfg = RunConfig(
            exchange_id=exchange,
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            costs=costs,
            bt=bt,
            use_tuning=False,
            wf={},
            mtf=["1h", "4h"],
            data_path=str(data_path),
        )

        # 백테스트 실행
        engine = BacktestEngine(runs_root=str(REPO_ROOT / "runs"))
        run_id = engine.run(df, cfg, theta=DEFAULT_THETA, scalping_mode=True, use_phase_accumulation=True)

        # 결과 로드
        run_dir = REPO_ROOT / "runs" / run_id
        equity_df = pd.read_parquet(run_dir / "equity.parquet")
        trades_df = pd.read_parquet(run_dir / "trades.parquet")

        # 성과 계산
        initial_equity = 1.0
        final_equity = equity_df['equity'].iloc[-1]
        total_return = ((final_equity - initial_equity) / initial_equity) * 100

        equity_series = equity_df['equity']
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # 거래 통계
        entry_trades = trades_df[trades_df['type'] == 'ENTRY']
        accum_trades = trades_df[trades_df['type'].isin(['ACCUM_LONG', 'ACCUM_SHORT'])]
        accum_tp = trades_df[trades_df['type'].isin(['ACCUM_LONG_TP', 'ACCUM_SHORT_TP'])]
        accum_stop = trades_df[trades_df['type'].isin(['ACCUM_LONG_STOP', 'ACCUM_SHORT_STOP'])]

        long_accum = len(trades_df[trades_df['type'] == 'ACCUM_LONG'])
        short_accum = len(trades_df[trades_df['type'] == 'ACCUM_SHORT'])

        results[symbol] = {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(entry_trades),
            'accum_trades': len(accum_trades),
            'long_accum': long_accum,
            'short_accum': short_accum,
            'accum_win': len(accum_tp),
            'accum_loss': len(accum_stop),
            'run_id': run_id
        }

        print(f"[{symbol}] 완료: 수익률 {total_return:+.2f}%, MDD {max_drawdown:.2f}%")

    # 최종 결과 출력
    print(f"\n{'='*70}")
    print(f"멀티 심볼 백테스팅 결과 요약")
    print(f"{'='*70}")
    print(f"{'심볼':<12} {'수익률':>10} {'MDD':>10} {'메인거래':>10} {'축적(롱/숏)':>15} {'축적승률':>10}")
    print(f"{'-'*70}")

    total_return_sum = 0
    for symbol, r in results.items():
        accum_total = r['accum_win'] + r['accum_loss']
        accum_winrate = r['accum_win'] / accum_total * 100 if accum_total > 0 else 0
        print(f"{symbol:<12} {r['total_return']:>+9.2f}% {r['max_drawdown']:>9.2f}% {r['total_trades']:>10d} {r['long_accum']:>6d}/{r['short_accum']:<6d} {accum_winrate:>9.1f}%")
        total_return_sum += r['total_return']

    print(f"{'-'*70}")
    print(f"{'합계':<12} {total_return_sum:>+9.2f}%")
    print(f"{'평균':<12} {total_return_sum/len(results):>+9.2f}%")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        # 멀티 심볼 테스트
        run_multi_symbol_test()
    else:
        # 단일 심볼 테스트
        main()
