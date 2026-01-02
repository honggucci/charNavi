#!/usr/bin/env python
"""
BTC 3개월 현물 백테스트 (15분봉)
- 현물 거래소의 spot 로직 사용
- BacktestEngine 사용
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
from wpcn.core.types import BacktestCosts, BacktestConfig, RunConfig
from wpcn.engine.backtest import BacktestEngine, DEFAULT_THETA

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


def main():
    print("=" * 80)
    print("BTC 3개월 현물 백테스트 (15분봉, BacktestEngine)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project Root: {PATHS.PROJECT_ROOT}")
    print("=" * 80)

    # Load configuration from environment
    symbol = os.getenv("SYMBOL", "BTC-USDT")
    timeframe = os.getenv("TIMEFRAME", "15m")
    days = int(os.getenv("BACKTEST_DAYS", "90"))

    print(f"\n[1/3] 데이터 로드 중...")
    print(f"  심볼: {symbol}")
    print(f"  타임프레임: {timeframe}")
    print(f"  기간: {days}일")

    df = load_recent_data(symbol, timeframe, days=days)

    if df is None or len(df) < 500:
        print(f"  Insufficient data, aborting")
        return

    start_date = df.index.min()
    end_date = df.index.max()
    actual_days = (end_date - start_date).days

    print(f"  Loaded {len(df):,} candles")
    print(f"  Period: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({actual_days} days)")

    # 백테스팅 설정
    print(f"\n[2/3] 백테스트 설정...")

    # Parse MTF timeframes from env
    mtf_timeframes = os.getenv("MTF_TIMEFRAMES", "1h,4h").split(",")

    costs = BacktestCosts(
        fee_bps=float(os.getenv("MAKER_FEE", "0.0002")) * 10000,  # Convert to bps
        slippage_bps=float(os.getenv("SLIPPAGE", "0.0002")) * 10000
    )
    bt = BacktestConfig(
        initial_equity=float(os.getenv("INITIAL_CAPITAL", "1.0")),
        max_hold_bars=int(os.getenv("MAX_HOLD_BARS", "30")),
        tp1_frac=float(os.getenv("TP1_FRAC", "0.5")),
        use_tp2=os.getenv("USE_TP2", "True").lower() == "true",
        conf_min=float(os.getenv("CONF_MIN", "0.50")),
        edge_min=float(os.getenv("EDGE_MIN", "0.60")),
        confirm_bars=int(os.getenv("CONFIRM_BARS", "1")),
        chaos_ret_atr=float(os.getenv("CHAOS_RET_ATR", "3.0")),
        adx_len=int(os.getenv("ADX_LEN", "14")),
        adx_trend=float(os.getenv("ADX_TREND", "15.0")),
    )

    cfg = RunConfig(
        exchange_id="binance",
        symbol=symbol.replace("-", "/"),
        timeframe=timeframe,
        days=days,
        costs=costs,
        bt=bt,
        use_tuning=False,
        wf={},
        mtf=mtf_timeframes,
        data_path=None,
    )

    print(f"  MTF: {', '.join(mtf_timeframes)}")
    print(f"  Phase A/B 축적: 활성화")
    print(f"  단타 모드: 활성화")

    # 백테스트 실행
    print(f"\n[3/3] 백테스트 실행 중...")

    try:
        scalping_mode = os.getenv("SCALPING_MODE", "True").lower() == "true"
        use_phase_accumulation = os.getenv("USE_PHASE_ACCUMULATION", "True").lower() == "true"

        engine = BacktestEngine(runs_root=str(PATHS.PROJECT_ROOT / "runs"))
        run_id = engine.run(
            df,
            cfg,
            theta=DEFAULT_THETA,
            scalping_mode=scalping_mode,
            use_phase_accumulation=use_phase_accumulation
        )
        print(f"  완료: {run_id}")

        # 결과 로드 및 출력
        run_dir = PATHS.PROJECT_ROOT / "runs" / run_id

        equity_df = pd.read_parquet(run_dir / "equity.parquet")
        trades_df = pd.read_parquet(run_dir / "trades.parquet")

        initial_equity = float(cfg.bt.initial_equity)
        final_equity = equity_df['equity'].iloc[-1]
        total_return = ((final_equity - initial_equity) / initial_equity) * 100

        equity_series = equity_df['equity']
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        entry_trades = trades_df[trades_df['type'] == 'ENTRY']
        num_trades = len(entry_trades)

        print(f"\n{'='*80}")
        print(f"백테스트 결과 - {symbol} (현물, 15분봉)")
        print(f"{'='*80}")
        print(f"  수익률: {total_return:.2f}%")
        print(f"  최대 낙폭 (MDD): {max_drawdown:.2f}%")
        print(f"  총 거래 수: {num_trades}")
        print(f"  결과 저장: {run_dir}")
        print(f"{'='*80}")

    except Exception as e:
        import traceback
        print(f"\n  ERROR: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
