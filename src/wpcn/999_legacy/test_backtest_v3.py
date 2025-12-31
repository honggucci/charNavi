"""
FuturesBacktestV3 테스트 스크립트
"""
import sys
sys.path.insert(0, r'c:\Users\gkghd\OneDrive\바탕 화면\coin_master\wpcn-backtester-cli-noflask\src')

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# 백테스트 엔진
from wpcn.execution.futures_backtest_v3 import FuturesBacktestV3, FuturesConfigV3
from wpcn.core.types import Theta

def load_5m_data(symbol: str = "BTC-USDT", days: int = 30) -> pd.DataFrame:
    """5분봉 데이터 로드"""
    base_path = Path(r"c:\Users\gkghd\OneDrive\바탕 화면\coin_master\wpcn-backtester-cli-noflask\data\bronze\binance\futures")
    data_path = base_path / symbol / "5m"

    if not data_path.exists():
        print(f"Data path not found: {data_path}")
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
                print(f"Error loading {month_file}: {e}")

    if not dfs:
        print("No data loaded")
        return None

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    # 최근 N일 데이터만
    end_date = df["timestamp"].max()
    start_date = end_date - timedelta(days=days)
    df = df[df["timestamp"] >= start_date]

    # 인덱스 설정
    df = df.set_index("timestamp")

    print(f"Loaded {len(df)} candles from {df.index.min()} to {df.index.max()}")
    return df


def main():
    print("=" * 60)
    print("FuturesBacktestV3 Test")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] Loading data...")
    df_5m = load_5m_data("BTC-USDT", days=14)  # 2주 데이터

    if df_5m is None or len(df_5m) < 100:
        print("Insufficient data for backtest")
        return

    # 설정
    print("\n[2] Setting up config...")
    config = FuturesConfigV3(
        leverage=10.0,
        risk_per_trade=0.01,  # 1%
        max_position_pct=0.2,  # 20%
        max_positions=2,
        max_daily_loss_pct=0.03,  # 3%
        use_dynamic_sl_tp=True,
        min_rr_ratio=1.5,
        tp_pct=0.02,  # 2%
        sl_pct=0.01,  # 1%
        max_hold_bars=48,  # 4시간
        min_score=3.5,  # 점수 기준 낮춤
        entry_cooldown=6  # 30분 쿨다운
    )

    # Theta 설정 (실제 필드에 맞게)
    theta = Theta(
        pivot_lr=5,       # 피벗 lookback/right
        box_L=20,         # 박스 길이
        m_freeze=3,       # 동결 기간
        atr_len=14,       # ATR 기간
        x_atr=1.5,        # ATR 배수
        m_bw=0.02,        # 박스 폭 배수
        N_reclaim=5,      # Reclaim 확인 봉수
        N_fill=2,         # 체결 확인 봉수
        F_min=0.5         # 최소 체결률
    )

    print(f"Config: leverage={config.leverage}x, risk={config.risk_per_trade*100}%")

    # 백테스트 실행
    print("\n[3] Running backtest...")
    engine = FuturesBacktestV3(config)

    try:
        result = engine.run(
            df_5m=df_5m,
            theta=theta,
            initial_capital=10000.0,
            save_to_db=False  # DB 저장 비활성화 (테스트)
        )

        # 결과 출력
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        stats = result['stats']
        print(f"\nCapital: ${stats['initial_capital']:,.0f} -> ${stats['final_equity']:,.2f}")
        print(f"Total Return: {stats['total_return']:.2f}%")
        print(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Avg P&L: ${stats['avg_pnl']:.2f}")
        print(f"Total Fees: ${stats['total_fees']:.2f}")
        print(f"Signals Generated: {stats['total_signals']}")
        print(f"Signals Executed: {stats['executed_signals']}")

        # 성과 지표
        if result['metrics']:
            metrics = result['metrics']
            print(f"\n--- Performance Metrics ---")
            print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
            print(f"Sortino Ratio: {metrics.sortino_ratio:.3f}")
            print(f"Calmar Ratio: {metrics.calmar_ratio:.3f}")

        # 거래 상세 (처음 10개만)
        trades_df = result['trades_df']
        if len(trades_df) > 0:
            print(f"\n--- Trade Details (First 10) ---")
            print(trades_df[['side', 'entry_price', 'exit_price', 'pnl', 'pnl_pct', 'exit_reason']].head(10).to_string())

        # 전략 분석
        print("\n[4] Running strategy analysis...")
        try:
            from pathlib import Path
            import importlib.util
            analyzer_path = Path(r"c:\Users\gkghd\OneDrive\바탕 화면\coin_master\wpcn-backtester-cli-noflask\src\wpcn\003_common\007_validation\strategy_analyzer.py")
            spec = importlib.util.spec_from_file_location("strategy_analyzer", analyzer_path)
            strategy_analyzer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_analyzer_module)
            StrategyAnalyzer = strategy_analyzer_module.StrategyAnalyzer

            analyzer = StrategyAnalyzer()
            analysis = analyzer.analyze(result['trades_df'], result['signals_df'])
            analyzer.print_report(analysis)
        except Exception as e:
            print(f"Strategy analysis failed: {e}")

        print("\nBacktest completed successfully!")

    except Exception as e:
        import traceback
        print(f"\nError during backtest: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
