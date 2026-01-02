"""
BTC 현물 백테스트 - MTF 점수 시스템 (STRATEGY_ARCHITECTURE.md 기반)

Original MTF 점수 기반 전략:
- 1W/1D/4H/1H/15M/5M 타임프레임 분석
- 점수 합산 (최소 4.0점 필요)
- TF 정렬 (최소 2개 타임프레임 일치)
- 손익비 (RR >= 1.5)
- Navigation Gate 통합
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
from wpcn._04_execution.broker_sim_mtf import simulate_mtf

# 튜닝 모듈 (조건부 import)
try:
    from wpcn._08_tuning.theta_space import ThetaSpace
    TUNING_AVAILABLE = True
except ImportError:
    TUNING_AVAILABLE = False

# 환경변수에서 튜닝 설정 로드
USE_TUNING = os.getenv('USE_TUNING', 'False').lower() == 'true'
TUNING_TRAIN_DAYS = int(os.getenv('TUNING_TRAIN_DAYS', '60'))
TUNING_TEST_DAYS = int(os.getenv('TUNING_TEST_DAYS', '30'))
TUNING_N_CANDIDATES = int(os.getenv('TUNING_N_CANDIDATES', '50'))

# 로깅 설정
logger = setup_logging(level="INFO")


def load_recent_data(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """최근 N일 데이터 로드"""
    # 15M은 5M로 리샘플링 (MTF 베이스 타임프레임)
    if timeframe == '15m':
        base_timeframe = '5m'
    else:
        base_timeframe = timeframe

    data_path = get_data_path("bronze", "binance", "futures", symbol, base_timeframe)

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


def run_walk_forward_tuning(
    df: pd.DataFrame,
    base_theta: Theta,
    costs: BacktestCosts,
    cfg: BacktestConfig,
    mtf_timeframes: list,
    spot_mode: bool,
    min_score: float,
    min_tf_alignment: int,
    min_rr_ratio: float,
    sl_atr_mult: float,
    tp_atr_mult: float,
    min_context_score: float,
    min_trigger_score: float,
    train_days: int = 60,
    test_days: int = 30,
    n_candidates: int = 50
) -> Theta:
    """
    Walk-Forward 튜닝 (0단계)

    훈련 기간에서 최적 Theta를 찾고,
    테스트 기간에서 검증 후 최종 Theta 반환

    Returns:
        최적화된 Theta
    """
    print(f"\n{'='*60}")
    print("[0단계] Walk-Forward 튜닝 시작")
    print(f"  훈련 기간: {train_days}일")
    print(f"  테스트 기간: {test_days}일")
    print(f"  후보 수: {n_candidates}")
    print(f"{'='*60}")

    if not TUNING_AVAILABLE:
        print("  [경고] ThetaSpace 모듈 사용 불가 - 기본 Theta 사용")
        return base_theta

    # ThetaSpace 초기화 (파라미터 범위 정의)
    theta_space = ThetaSpace(
        pivot_lr=(2, 5),
        box_L=(30, 100),
        m_freeze=(8, 32),
        atr_len=(10, 20),
        x_atr=(1.5, 3.0),
        m_bw=(0.01, 0.05),
        N_reclaim=(4, 16),
        N_fill=(3, 10),
        F_min=(0.2, 0.5)
    )

    # 데이터 분할 (train/test)
    bars_per_day = 288  # 5분봉 기준 하루 288개
    train_bars = train_days * bars_per_day
    test_bars = test_days * bars_per_day
    embargo_bars = bars_per_day  # 1일 embargo

    total_required = train_bars + embargo_bars + test_bars
    if len(df) < total_required:
        print(f"  [경고] 데이터 부족 ({len(df)} < {total_required}) - 기본 Theta 사용")
        return base_theta

    # 단일 fold (최근 데이터 기준)
    test_end = len(df)
    test_start = test_end - test_bars
    train_end = test_start - embargo_bars
    train_start = train_end - train_bars

    train_df = df.iloc[train_start:train_end]
    test_df = df.iloc[test_start:test_end]

    print(f"  훈련 데이터: {len(train_df):,}봉")
    print(f"  테스트 데이터: {len(test_df):,}봉")

    # 후보 탐색
    best_theta = base_theta
    best_score = -float('inf')

    for i in range(n_candidates):
        try:
            # 랜덤 Theta 샘플링
            candidate_theta = theta_space.sample()

            # 훈련 데이터에서 백테스트
            equity_df, trades_df, _, _ = simulate_mtf(
                df=train_df,
                theta=candidate_theta,
                costs=costs,
                cfg=cfg,
                mtf=mtf_timeframes,
                spot_mode=spot_mode,
                min_score=min_score,
                min_tf_alignment=min_tf_alignment,
                min_rr_ratio=min_rr_ratio,
                sl_atr_mult=sl_atr_mult,
                tp_atr_mult=tp_atr_mult,
                min_context_score=min_context_score,
                min_trigger_score=min_trigger_score
            )

            # 목적함수: 수익률 - MDD
            if len(trades_df) == 0:
                continue

            final_equity = equity_df['equity'].iloc[-1]
            initial_equity = cfg.initial_equity
            ret_pct = (final_equity - initial_equity) / initial_equity * 100
            mdd = ((equity_df['equity'].cummax() - equity_df['equity']) / equity_df['equity'].cummax()).max() * 100

            score = ret_pct - mdd

            if score > best_score:
                best_score = score
                best_theta = candidate_theta
                print(f"    [{i+1}/{n_candidates}] 새 최고 점수: {score:.2f} (수익률: {ret_pct:.2f}%, MDD: {mdd:.2f}%)")

        except Exception as e:
            continue

        # 진행 상황 (10개마다)
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{n_candidates}] 탐색 중...")

    # 테스트 데이터에서 검증
    print(f"\n  === 테스트 검증 ===")
    try:
        equity_df, trades_df, _, _ = simulate_mtf(
            df=test_df,
            theta=best_theta,
            costs=costs,
            cfg=cfg,
            mtf=mtf_timeframes,
            spot_mode=spot_mode,
            min_score=min_score,
            min_tf_alignment=min_tf_alignment,
            min_rr_ratio=min_rr_ratio,
            sl_atr_mult=sl_atr_mult,
            tp_atr_mult=tp_atr_mult,
            min_context_score=min_context_score,
            min_trigger_score=min_trigger_score
        )

        if len(trades_df) > 0:
            final_equity = equity_df['equity'].iloc[-1]
            ret_pct = (final_equity - cfg.initial_equity) / cfg.initial_equity * 100
            mdd = ((equity_df['equity'].cummax() - equity_df['equity']) / equity_df['equity'].cummax()).max() * 100
            print(f"  테스트 수익률: {ret_pct:.2f}%")
            print(f"  테스트 MDD: {mdd:.2f}%")
        else:
            print(f"  테스트 거래 없음")

    except Exception as e:
        print(f"  테스트 오류: {e}")

    print(f"\n  최적 Theta:")
    print(f"    pivot_lr: {best_theta.pivot_lr}")
    print(f"    box_L: {best_theta.box_L}")
    print(f"    m_freeze: {best_theta.m_freeze}")
    print(f"    N_fill: {best_theta.N_fill}")
    print(f"{'='*60}")

    return best_theta


def main():
    print("=" * 80)
    print("BTC 현물 백테스트 - MTF 점수 시스템")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project Root: {PATHS.PROJECT_ROOT}")
    print(f"Data Dir: {PATHS.DATA_DIR}")
    print("=" * 80)
    print("\nMTF 점수 시스템 (STRATEGY_ARCHITECTURE.md):")
    print("  - 1W: Weekly trend (EMA 50/200)")
    print("  - 1D: Daily trend (EMA 50/200)")
    print("  - 4H: Trend filter (BULLISH/BEARISH)")
    print("  - 1H: Wyckoff structure (ACCUMULATION/RE-ACCUMULATION/etc)")
    print("  - 15M: RSI divergence detection")
    print("  - 5M: Entry timing (reversal candles, oversold/overbought)")
    print("")
    print("  신호 조건:")
    print("    - 최소 점수: 4.0")
    print("    - TF 정렬: >= 2 timeframes")
    print("    - 손익비 (RR): >= 1.5")
    print("=" * 80)

    # Load configuration from environment
    symbol = os.getenv("SYMBOL", "BTC-USDT")
    timeframe = os.getenv("TIMEFRAME", "15m")
    days = int(os.getenv("BACKTEST_DAYS", "90"))

    # Theta (box detection parameters)
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
        max_hold_bars=int(os.getenv("MAX_HOLD_BARS", "288")),  # 3 days for swing trades
        tp1_frac=float(os.getenv("TP1_FRAC", "0.5")),
        use_tp2=os.getenv("USE_TP2", "True").lower() == "true",
        conf_min=float(os.getenv("CONF_MIN", "0.55")),  # 낮춤: MTF 자체 필터링 사용
        edge_min=float(os.getenv("EDGE_MIN", "0.0")),   # 0으로: MTF는 자체 Trigger Score 사용
        confirm_bars=int(os.getenv("CONFIRM_BARS", "1")),
        chaos_ret_atr=float(os.getenv("CHAOS_RET_ATR", "3.0")),
        adx_len=int(os.getenv("ADX_LEN", "14")),
        adx_trend=float(os.getenv("ADX_TREND", "15.0"))
    )

    # MTF scoring parameters
    min_score = float(os.getenv("MIN_SCORE", "4.0"))
    min_tf_alignment = int(os.getenv("MIN_TF_ALIGNMENT", "2"))
    min_rr_ratio = float(os.getenv("MIN_RR_RATIO", "1.5"))
    sl_atr_mult = float(os.getenv("SL_ATR_MULT", "1.5"))
    tp_atr_mult = float(os.getenv("TP_ATR_MULT", "2.5"))

    # V2: Context/Trigger 파라미터
    min_context_score = float(os.getenv("MIN_CONTEXT_SCORE", "2.0"))
    min_trigger_score = float(os.getenv("MIN_TRIGGER_SCORE", "1.5"))

    print(f"\n설정:")
    print(f"  심볼: {symbol}")
    print(f"  타임프레임: {timeframe} (베이스: 5m)")
    print(f"  초기 자본: ${cfg.initial_equity:,.0f}")
    print(f"  백테스트 기간: 최근 {days}일")
    print(f"  최소 점수: {min_score}")
    print(f"  최소 TF 정렬: {min_tf_alignment}")
    print(f"  최소 손익비: {min_rr_ratio}")
    print(f"  SL ATR 배수: {sl_atr_mult}")
    print(f"  TP ATR 배수: {tp_atr_mult}")
    print(f"  최소 Context 점수: {min_context_score}")
    print(f"  최소 Trigger 점수: {min_trigger_score}")

    # 백테스트 실행
    print(f"\n{'='*60}")
    print(f"[{symbol}] Loading {days} days data...")

    df = load_recent_data(symbol, timeframe, days=days)

    if df is None or len(df) < 1000:
        print(f"  Insufficient data, aborting (need at least 1000 bars for 5M)")
        return

    # 데이터 기간 정보
    start_date = df.index.min()
    end_date = df.index.max()
    actual_days = (end_date - start_date).days

    print(f"  Loaded {len(df):,} candles (5M)")
    print(f"  Period: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({actual_days} days)")
    print(f"  Running MTF scoring backtest...")

    # MTF timeframes
    mtf_timeframes = ['15m', '1h', '4h', '1d', '1w']
    spot_mode = True  # 현물 모드 (숏 비활성화)

    # === 0단계: Walk-Forward 튜닝 (옵션) ===
    if USE_TUNING and TUNING_AVAILABLE:
        print(f"\n[튜닝 모드 활성화]")
        theta = run_walk_forward_tuning(
            df=df,
            base_theta=theta,
            costs=costs,
            cfg=cfg,
            mtf_timeframes=mtf_timeframes,
            spot_mode=spot_mode,
            min_score=min_score,
            min_tf_alignment=min_tf_alignment,
            min_rr_ratio=min_rr_ratio,
            sl_atr_mult=sl_atr_mult,
            tp_atr_mult=tp_atr_mult,
            min_context_score=min_context_score,
            min_trigger_score=min_trigger_score,
            train_days=TUNING_TRAIN_DAYS,
            test_days=TUNING_TEST_DAYS,
            n_candidates=TUNING_N_CANDIDATES
        )
    elif USE_TUNING and not TUNING_AVAILABLE:
        print(f"\n[경고] USE_TUNING=True이나 ThetaSpace 모듈 사용 불가")

    # MTF 백테스트 실행
    try:
        equity_df, trades_df, signals_df, nav_df = simulate_mtf(
            df=df,
            theta=theta,
            costs=costs,
            cfg=cfg,
            mtf=mtf_timeframes,
            spot_mode=spot_mode,
            min_score=min_score,
            min_tf_alignment=min_tf_alignment,
            min_rr_ratio=min_rr_ratio,
            sl_atr_mult=sl_atr_mult,
            tp_atr_mult=tp_atr_mult,
            atr_period=14,
            min_context_score=min_context_score,
            min_trigger_score=min_trigger_score
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
            'TP1', 'TP2', 'STOP', 'TIME_EXIT'
        ])]

        if len(exit_trades) > 0:
            wins = exit_trades[exit_trades['pnl_pct'] > 0]
            losses = exit_trades[exit_trades['pnl_pct'] <= 0]
            win_rate = len(wins) / len(exit_trades) * 100 if len(exit_trades) > 0 else 0
            avg_pnl_pct = exit_trades['pnl_pct'].mean()
            profit_factor = wins['pnl_pct'].sum() / abs(losses['pnl_pct'].sum()) if len(losses) > 0 and losses['pnl_pct'].sum() != 0 else 0
        else:
            win_rate = 0
            avg_pnl_pct = 0
            profit_factor = 0

        # Entry trades count
        entry_trades = trades_df[trades_df['type'] == 'ENTRY']
        long_trades = len(entry_trades[entry_trades['side'] == 'long'])
        short_trades = len(entry_trades[entry_trades['side'] == 'short'])

        print(f"\n{'='*80}")
        print(f"백테스트 결과 - {symbol} (MTF 점수 시스템)")
        print(f"{'='*80}")
        print(f"  초기 자본: ${initial_capital:,.0f}")
        print(f"  최종 자본: ${final_equity:,.2f}")
        print(f"  수익률: {total_return:.2f}%")
        print(f"  최대 낙폭 (MDD): {max_dd:.2f}%")
        print(f"  총 진입: {len(entry_trades)} (롱: {long_trades}, 숏: {short_trades})")
        print(f"  총 청산: {len(exit_trades)}")
        print(f"  승률: {win_rate:.1f}%")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print(f"  평균 손익: {avg_pnl_pct:.2f}%")
        print(f"  총 신호: {len(signals_df)}")

        if len(entry_trades) > 0:
            avg_score = entry_trades['score'].mean()
            avg_rr = entry_trades['rr_ratio'].mean()
            print(f"\n  평균 신호 점수: {avg_score:.2f}")
            print(f"  평균 손익비: {avg_rr:.2f}")

            # V2: Context/Trigger 및 MFE/MAE 통계
            if 'context_score' in entry_trades.columns:
                avg_context = entry_trades['context_score'].mean()
                avg_trigger = entry_trades['trigger_score'].mean()
                print(f"  평균 Context 점수: {avg_context:.2f}")
                print(f"  평균 Trigger 점수: {avg_trigger:.2f}")

            if 'mfe' in exit_trades.columns:
                avg_mfe = exit_trades['mfe'].mean()
                avg_mae = exit_trades['mae'].mean()
                print(f"\n  === MFE/MAE 분석 ===")
                print(f"  평균 MFE (최대 유리 이동): {avg_mfe:.2f}%")
                print(f"  평균 MAE (최대 불리 이동): {avg_mae:.2f}%")

                # Phase별 통계
                if 'phase' in exit_trades.columns:
                    print(f"\n  === Phase별 분석 ===")
                    phase_stats = exit_trades.groupby('phase').agg({
                        'pnl_pct': ['count', 'mean'],
                        'mfe': 'mean',
                        'mae': 'mean'
                    }).round(2)
                    print(phase_stats.to_string())

        # Save results for analysis
        if len(trades_df) > 0:
            output_dir = Path("runs") / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_mtf_backtest"
            output_dir.mkdir(parents=True, exist_ok=True)

            trades_df.to_parquet(output_dir / "trades.parquet")
            equity_df.to_parquet(output_dir / "equity.parquet")
            signals_df.to_parquet(output_dir / "signals.parquet")
            nav_df.to_parquet(output_dir / "nav.parquet")
            df.to_parquet(output_dir / "candles.parquet")

            print(f"\n  Results saved to: {output_dir}")

        print(f"\n{'='*80}")
        print("백테스트 완료!")
        print(f"{'='*80}")

    except Exception as e:
        import traceback
        print(f"\n  ERROR: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
