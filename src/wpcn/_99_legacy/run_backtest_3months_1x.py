"""
BTC 3개월 백테스트 (레버리지 1배)
- 선물 시장이지만 1배 레버리지로 현물과 동일한 조건
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from datetime import datetime, timedelta
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

from wpcn._00_config.config import PATHS, get_data_path, get_results_path, setup_logging
from wpcn._04_execution.futures_backtest_v8 import V8BacktestEngine, V8Config

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


def run_backtest_v8(symbol: str, df: pd.DataFrame, config: V8Config) -> dict:
    """V8 백테스트 실행"""
    engine = V8BacktestEngine(config)

    try:
        # V8 엔진은 df_5m만 받지만, 실제로는 15분봉 데이터를 사용
        result = engine.run(
            df_5m=df,
            symbol=symbol,
            initial_capital=10000.0
        )
        return result
    except Exception as e:
        import traceback
        logger.error(f"Backtest failed for {symbol}: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}


def main():
    print("=" * 80)
    print("BTC 3개월 백테스트 - 레버리지 1배 (현물 시뮬레이션)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project Root: {PATHS.PROJECT_ROOT}")
    print(f"Data Dir: {PATHS.DATA_DIR}")
    print("=" * 80)
    print("\nTimeframe Structure:")
    print("  - 4H: Wyckoff structure")
    print("  - 1H: Wyckoff structure (Spring/Upthrust)")
    print("  - 15M: Stoch RSI FILTER (oversold/overbought)")
    print("  - 5M: RSI Divergence + ZigZag + Entry timing")
    print("=" * 80)

    # Load configuration from environment
    symbol = os.getenv("SYMBOL", "BTC-USDT")
    timeframe = os.getenv("TIMEFRAME", "15m")
    days = int(os.getenv("BACKTEST_DAYS", "90"))

    # 설정 - Environment variables
    config = V8Config(
        leverage=float(os.getenv("LEVERAGE", "1.0")),
        position_pct=float(os.getenv("POSITION_PCT", "0.95")),
        sl_atr_mult=float(os.getenv("SL_ATR_MULT", "1.5")),
        tp_atr_mult=float(os.getenv("TP_ATR_MULT", "2.5")),
        min_score=float(os.getenv("MIN_SCORE", "5.0")),
        min_tf_alignment=int(os.getenv("MIN_TF_ALIGNMENT", "3")),
        min_rr_ratio=float(os.getenv("MIN_RR_RATIO", "1.5")),
        warmup_bars=int(os.getenv("WARMUP_BARS", "500")),
        cooldown_bars=int(os.getenv("COOLDOWN_BARS", "24")),
        max_hold_bars=int(os.getenv("MAX_HOLD_BARS", "72")),
        box_lookback=int(os.getenv("BOX_LOOKBACK", "50")),
        spring_threshold=float(os.getenv("SPRING_THRESHOLD", "0.005")),
        rsi_period=int(os.getenv("RSI_PERIOD", "14")),
        stoch_rsi_period=int(os.getenv("STOCH_RSI_PERIOD", "14")),
        div_lookback=int(os.getenv("DIV_LOOKBACK", "20")),
        zigzag_threshold=float(os.getenv("ZIGZAG_THRESHOLD", "0.02")),
        maker_fee=float(os.getenv("MAKER_FEE", "0.0002")),
        taker_fee=float(os.getenv("TAKER_FEE", "0.0004")),
        slippage=float(os.getenv("SLIPPAGE", "0.0002")),
        use_probability_model=os.getenv("USE_PROBABILITY_MODEL", "False").lower() == "true",
        min_tp_probability=float(os.getenv("MIN_TP_PROBABILITY", "0.50")),
        min_ev_r=float(os.getenv("MIN_EV_R", "0.05")),
        min_expected_rr=float(os.getenv("MIN_EXPECTED_RR", "0.003")),
        # Navigation Gate
        use_navigation_gate=os.getenv("USE_NAVIGATION_GATE", "True").lower() == "true",
        conf_min=float(os.getenv("CONF_MIN", "0.50")),
        edge_min=float(os.getenv("EDGE_MIN", "0.60")),
        adx_trend=float(os.getenv("ADX_TREND", "15.0")),
        chaos_ret_atr=float(os.getenv("CHAOS_RET_ATR", "3.0"))
    )

    print(f"\n설정:")
    print(f"  심볼: {symbol}")
    print(f"  타임프레임: {timeframe}")
    print(f"  레버리지: {config.leverage}x")
    print(f"  포지션 크기: {config.position_pct*100}%")
    print(f"  백테스트 기간: 최근 {days}일")

    # 백테스트 실행
    print(f"\n{'='*60}")
    print(f"[{symbol}] Loading {days} days data ({timeframe} candles)...")

    df = load_recent_data(symbol, timeframe, days=days)

    if df is None or len(df) < 2000:
        print(f"  Insufficient data, aborting")
        return

    # 데이터 기간 정보
    start_date = df.index.min()
    end_date = df.index.max()
    days = (end_date - start_date).days

    print(f"  Loaded {len(df):,} candles")
    print(f"  Period: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({days} days)")
    print(f"  Running V8 backtest...")

    result = run_backtest_v8(symbol, df, config)

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        if "traceback" in result:
            print(f"  {result['traceback']}")
        return

    stats = result["stats"]

    print(f"\n{'='*80}")
    print(f"백테스트 결과 - {symbol} (레버리지 {config.leverage}x)")
    print(f"{'='*80}")
    print(f"  수익률: {stats['total_return']:.2f}%")
    print(f"  최대 낙폭 (MDD): {stats['max_drawdown']:.2f}%")
    print(f"  총 거래 수: {stats['total_trades']} (롱: {stats['long_trades']}, 숏: {stats['short_trades']})")
    print(f"  승률: {stats['win_rate']:.1f}%")
    print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    print(f"  평균 손익: {stats['avg_pnl']:.2f}%")
    print(f"  총 신호: {stats['total_signals']}")

    if 'tp_exits' in stats:
        print(f"\n  청산 방식:")
        print(f"    - 익절 (TP): {stats['tp_exits']}")
        print(f"    - 손절 (SL): {stats['sl_exits']}")
        print(f"    - 타임아웃: {stats['timeout_exits']}")
        print(f"    - 청산: {stats['liquidations']}")

    print(f"\n{'='*80}")
    print("백테스트 완료!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
