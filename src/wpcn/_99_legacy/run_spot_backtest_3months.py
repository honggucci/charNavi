"""
BTC 3개월 현물 백테스트 (broker_sim 사용)
- 현물 시장: 숏 불가능, 롱만 가능
- 레버리지 없음 (1배 고정)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from datetime import datetime
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

from wpcn._00_config.config import PATHS, get_data_path, setup_logging
from wpcn._03_common._01_core.types import Theta, BacktestCosts, BacktestConfig
from wpcn._04_execution.broker_sim import simulate

# 로깅 설정
logger = setup_logging(level="INFO")


def load_recent_data(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """최근 N일 데이터 로드"""
    data_path = get_data_path("bronze", "binance", "futures", symbol, timeframe)

    if not data_path.exists():
        logger.warning(f"Data path not found: {data_path}")
        return None

    dfs = []
    for year_dir in data_path.iterdir():
        if not year_dir.is_dir():
            continue
        for month_file in year_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(month_file)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {month_file}: {e}")

    if not dfs:
        logger.warning(f"No data files found for {symbol}")
        return None

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    # 최근 N일만 필터링
    cutoff_date = df.index.max() - pd.Timedelta(days=days)
    df = df[df.index >= cutoff_date]

    return df


def load_multi_timeframe_data(symbol: str, days: int) -> tuple:
    """
    멀티타임프레임 데이터 로드 (5분봉 + 15분봉)

    Returns:
        (df_5m, df_15m): 5분봉, 15분봉 DataFrame
    """
    print(f"\n{'='*60}")
    print("Loading Multi-Timeframe Data...")

    # 5분봉 로드
    df_5m = load_recent_data(symbol, "5m", days)
    if df_5m is None or len(df_5m) < 500:
        logger.error(f"Insufficient 5m data for {symbol}")
        return None, None

    # 15분봉 로드
    df_15m = load_recent_data(symbol, "15m", days)
    if df_15m is None or len(df_15m) < 500:
        logger.error(f"Insufficient 15m data for {symbol}")
        return None, None

    print(f"  5M: {len(df_5m):,} candles ({df_5m.index.min()} ~ {df_5m.index.max()})")
    print(f"  15M: {len(df_15m):,} candles ({df_15m.index.min()} ~ {df_15m.index.max()})")
    print(f"{'='*60}")

    return df_5m, df_15m


def main():
    print("=" * 80)
    print("BTC 3개월 현물 백테스트 (broker_sim - MTF)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project Root: {PATHS.PROJECT_ROOT}")
    print(f"Data Dir: {PATHS.DATA_DIR}")
    print("=" * 80)
    print("\n현물 거래 특징:")
    print("  - 롱만 가능 (숏 불가능)")
    print("  - 레버리지 없음 (1배 고정)")
    print("  - MTF 신호 분석: 15M Stoch RSI + 5M/15M RSI Divergence")
    print("=" * 80)

    # Load configuration from environment
    symbol = os.getenv("SYMBOL", "BTC-USDT")
    days = int(os.getenv("BACKTEST_DAYS", "90"))

    # 현물 설정 - Theta (box detection parameters)
    theta = Theta(
        pivot_lr=3,
        box_L=int(os.getenv("BOX_LOOKBACK", "50")),
        m_freeze=int(os.getenv("M_FREEZE", "16")),
        atr_len=14,
        x_atr=2.0,
        m_bw=0.02,
        N_reclaim=8,
        N_fill=int(os.getenv("N_FILL", "5")),
        F_min=0.3
    )

    costs = BacktestCosts(
        fee_bps=float(os.getenv("MAKER_FEE", "0.0002")) * 10000,  # Convert to bps
        slippage_bps=float(os.getenv("SLIPPAGE", "0.0002")) * 10000
    )

    cfg = BacktestConfig(
        initial_equity=float(os.getenv("INITIAL_CAPITAL", "10000.0")),
        max_hold_bars=int(os.getenv("MAX_HOLD_BARS", "72")),
        tp1_frac=float(os.getenv("TP1_FRAC", "0.5")),
        use_tp2=os.getenv("USE_TP2", "True").lower() == "true",
        conf_min=float(os.getenv("CONF_MIN", "0.50")),
        edge_min=float(os.getenv("EDGE_MIN", "0.60")),
        confirm_bars=int(os.getenv("CONFIRM_BARS", "1")),
        chaos_ret_atr=float(os.getenv("CHAOS_RET_ATR", "3.0")),
        adx_len=int(os.getenv("ADX_LEN", "14")),
        adx_trend=float(os.getenv("ADX_TREND", "15.0"))
    )

    print(f"\n설정:")
    print(f"  심볼: {symbol}")
    print(f"  메인 타임프레임: 15M")
    print(f"  보조 타임프레임: 5M")
    print(f"  초기 자본: ${cfg.initial_equity:,.0f}")
    print(f"  백테스트 기간: 최근 {days}일")

    # 백테스트 실행: MTF 데이터 로드
    df_5m, df_15m = load_multi_timeframe_data(symbol, days)

    if df_5m is None or df_15m is None:
        print(f"  Data loading failed, aborting")
        return

    # 메인 타임프레임은 15분봉
    df = df_15m

    # 데이터 기간 정보
    start_date = df.index.min()
    end_date = df.index.max()
    days = (end_date - start_date).days

    print(f"  Loaded {len(df):,} candles")
    print(f"  Period: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({days} days)")
    print(f"  Running spot backtest (broker_sim)...")

    # Parse MTF timeframes from env
    mtf_str = os.getenv("MTF_TIMEFRAMES", "1h,4h")
    mtf_timeframes = mtf_str.split(",") if mtf_str else None

    scalping_mode = os.getenv("SCALPING_MODE", "False").lower() == "true"
    use_phase_accumulation = os.getenv("USE_PHASE_ACCUMULATION", "True").lower() == "true"
    spot_mode = os.getenv("SPOT_MODE", "True").lower() == "true"

    # 현물 백테스트 실행 (MTF 데이터 전달)
    try:
        equity_df, trades_df, signals_df, nav_df = simulate(
            df=df,  # 15분봉 (메인)
            df_5m=df_5m,  # 5분봉 (보조)
            theta=theta,
            costs=costs,
            cfg=cfg,
            mtf=mtf_timeframes,
            scalping_mode=scalping_mode,
            use_phase_accumulation=use_phase_accumulation,
            spot_mode=spot_mode
        )

        # 통계 계산
        if len(trades_df) == 0:
            print(f"\n  No trades executed")
            return

        initial_capital = cfg.initial_equity
        final_equity = equity_df['equity'].iloc[-1]
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
        max_dd = ((equity_df['equity'].cummax() - equity_df['equity']) / equity_df['equity'].cummax()).max() * 100

        # Exit trades (TP/SL) have pnl_pct
        exit_trades = trades_df[trades_df['type'].isin([
            'TP1_MTF', 'TP2_MTF', 'STOP_MTF', 'TIME_EXIT_MTF',  # MTF spot exit types
            'ACCUM_TP1', 'ACCUM_TP2', 'ACCUM_SL',  # Legacy accumulation types (if enabled)
            'ACCUM_SHORT_TP1', 'ACCUM_SHORT_TP2', 'ACCUM_SHORT_SL',
        ])]

        if len(exit_trades) > 0:
            wins = exit_trades[exit_trades['pnl_pct'] > 0]
            losses = exit_trades[exit_trades['pnl_pct'] <= 0]
            win_rate = len(wins) / len(exit_trades) * 100 if len(exit_trades) > 0 else 0
            avg_pnl_pct = exit_trades['pnl_pct'].mean()
        else:
            wins = pd.DataFrame()
            losses = pd.DataFrame()
            win_rate = 0
            avg_pnl_pct = 0

        # Entry trades count
        entry_trades = trades_df[trades_df['type'].isin([
            'ENTRY_MTF_SPOT',  # MTF spot entry type
            'ACCUM_LONG', 'ACCUM_SHORT'  # Legacy types (if enabled)
        ])]
        long_trades = len(entry_trades[entry_trades['side'] == 'long'])
        short_trades = len(entry_trades[entry_trades['side'] == 'short'])

        print(f"\n{'='*80}")
        print(f"백테스트 결과 - {symbol} (현물)")
        print(f"{'='*80}")
        print(f"  초기 자본: ${initial_capital:,.0f}")
        print(f"  최종 자본: ${final_equity:,.2f}")
        print(f"  수익률: {total_return:.2f}%")
        print(f"  최대 낙폭 (MDD): {max_dd:.2f}%")
        print(f"  총 진입: {len(entry_trades)} (롱: {long_trades}, 숏: {short_trades})")
        print(f"  총 청산: {len(exit_trades)}")
        print(f"  승률: {win_rate:.1f}%")
        print(f"  평균 손익: {avg_pnl_pct:.2f}%")
        print(f"  총 신호: {len(signals_df)}")

        # Save trades to CSV for analysis
        if len(trades_df) > 0:
            trades_df.to_csv("trades_debug.csv", index=False)
            print(f"\n  Trades saved to trades_debug.csv ({len(trades_df)} total trades)")

        print(f"\n{'='*80}")
        print("백테스트 완료!")
        print(f"{'='*80}")

    except Exception as e:
        import traceback
        print(f"\n  ERROR: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
