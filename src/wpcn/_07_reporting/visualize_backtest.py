"""
Backtest Results Visualizer using TradingView-style charts

Usage:
    python visualize_backtest.py
    python visualize_backtest.py --run runs/2025-12-31_19-31-06_binance_BTC-USDT_15m
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

try:
    from lightweight_charts import Chart
except ImportError:
    print("[ERROR] lightweight-charts not installed!")
    print("Run: pip install lightweight-charts")
    exit(1)


def get_latest_run(runs_dir: Path) -> Path:
    """Get the most recent backtest run directory"""
    runs = list(runs_dir.glob('*'))
    if not runs:
        raise FileNotFoundError(f"No runs found in {runs_dir}")

    # Sort by directory name (contains timestamp)
    latest = sorted(runs, reverse=True)[0]
    return latest


def load_backtest_data(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load backtest results from parquet files"""

    print(f"[INFO] Loading data from: {run_dir.name}")

    # Load candles
    candles_path = run_dir / 'candles.parquet'
    if not candles_path.exists():
        raise FileNotFoundError(f"Candles file not found: {candles_path}")
    candles = pd.read_parquet(candles_path)

    # Load trades
    trades_path = run_dir / 'trades.parquet'
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades file not found: {trades_path}")
    trades = pd.read_parquet(trades_path)

    # Load equity (optional)
    equity_path = run_dir / 'equity.parquet'
    equity = pd.read_parquet(equity_path) if equity_path.exists() else None

    print(f"  Candles: {len(candles):,}")
    print(f"  Trades: {len(trades):,}")
    if equity is not None:
        print(f"  Equity points: {len(equity):,}")

    return candles, trades, equity


def visualize_backtest(run_dir: Path, show_equity: bool = False):
    """
    Visualize backtest results with TradingView-style interactive chart

    Args:
        run_dir: Path to backtest run directory
        show_equity: Whether to show equity curve in separate window
    """

    # Load data
    candles, trades, equity = load_backtest_data(run_dir)

    # Prepare candle data
    # lightweight-charts expects columns: time, open, high, low, close, volume
    candles_data = candles.reset_index()

    # Rename index column to 'time' if needed
    if candles_data.columns[0] != 'time':
        candles_data.rename(columns={candles_data.columns[0]: 'time'}, inplace=True)

    # Ensure datetime is in correct format
    if not pd.api.types.is_datetime64_any_dtype(candles_data['time']):
        candles_data['time'] = pd.to_datetime(candles_data['time'])

    print(f"\n[INFO] Creating TradingView-style chart...")

    # Create main price chart
    chart = Chart(
        width=1600,
        height=900,
        inner_width=1.0,
        inner_height=1.0
    )

    # Set candle data
    chart.set(candles_data)

    # Add trade markers
    entry_trades = trades[trades['type'] == 'ENTRY']
    # Exit types: TP1, TP2, STOP, TIME_EXIT, ACCUM_TP
    exit_types = ['TP1', 'TP2', 'STOP', 'TIME_EXIT', 'ACCUM_TP']
    exit_trades = trades[trades['type'].isin(exit_types)]

    print(f"  Entry signals: {len(entry_trades)}")
    print(f"  Exit signals: {len(exit_trades)}")

    # Entry markers (buy)
    for idx, trade in entry_trades.iterrows():
        trade_time = trade['time']
        if not isinstance(trade_time, (pd.Timestamp, datetime)):
            trade_time = pd.to_datetime(trade_time)

        chart.marker(
            time=trade_time,
            position='below',
            color='#26a69a',  # Green
            shape='arrow_up',
            text=f"BUY @ ${trade['price']:.0f}"
        )

    # Exit markers (sell)
    for idx, trade in exit_trades.iterrows():
        trade_time = trade['time']
        if not isinstance(trade_time, (pd.Timestamp, datetime)):
            trade_time = pd.to_datetime(trade_time)

        # Get exit type and PnL
        exit_type = trade['type']
        pnl = trade.get('pnl', 0)
        pnl_pct = trade.get('pnl_pct', 0)

        # Color and label based on exit type
        if exit_type in ['TP1', 'TP2', 'ACCUM_TP']:
            color = '#26a69a'  # Green for take profit
            label = 'TP'
        elif exit_type == 'STOP':
            color = '#ef5350'  # Red for stop loss
            label = 'SL'
        else:  # TIME_EXIT
            color = '#ffeb3b'  # Yellow for time exit
            label = 'TIME'

        text = f"{label} @ ${trade['price']:.0f}"
        if pnl_pct != 0:
            text += f" ({pnl_pct:+.1f}%)"

        chart.marker(
            time=trade_time,
            position='above',
            color=color,
            shape='arrow_down',
            text=text
        )

    # Print summary
    print(f"\n[SUMMARY]")
    print(f"  Period: {candles.index[0]} -> {candles.index[-1]}")
    print(f"  Total Trades: {len(trades)}")

    if equity is not None and len(equity) > 0:
        initial_equity = equity['equity'].iloc[0]
        final_equity = equity['equity'].iloc[-1]
        total_return = (final_equity / initial_equity - 1) * 100
        print(f"  Initial Capital: ${initial_equity:,.2f}")
        print(f"  Final Equity: ${final_equity:,.2f}")
        print(f"  Total Return: {total_return:+.2f}%")

    # Calculate win rate
    if len(exit_trades) > 0:
        profitable_trades = ((exit_trades.get('pnl', 0) > 0) |
                           (exit_trades.get('pnl_pct', 0) > 0)).sum()
        win_rate = profitable_trades / len(exit_trades) * 100
        print(f"  Win Rate: {win_rate:.1f}%")

    print(f"\n[SUCCESS] Opening chart in browser...")
    chart.show()

    # Show equity curve in separate window if requested
    if show_equity and equity is not None:
        print(f"\n[INFO] Creating equity curve...")
        equity_chart = Chart(
            width=1600,
            height=400,
            inner_width=1.0,
            inner_height=1.0
        )

        equity_data = equity.reset_index()
        if equity_data.columns[0] != 'time':
            equity_data.rename(columns={equity_data.columns[0]: 'time'}, inplace=True)

        # Create line chart for equity
        equity_chart.set(equity_data[['time', 'equity']])
        equity_chart.show()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Visualize backtest results with TradingView-style charts'
    )
    parser.add_argument(
        '--run',
        type=str,
        default=None,
        help='Path to backtest run directory (default: latest run in runs/)'
    )
    parser.add_argument(
        '--equity',
        action='store_true',
        help='Show equity curve in separate window'
    )

    args = parser.parse_args()

    # Determine run directory
    if args.run:
        run_dir = Path(args.run)
    else:
        runs_dir = Path('runs')
        if not runs_dir.exists():
            print(f"[ERROR] Runs directory not found: {runs_dir}")
            exit(1)
        run_dir = get_latest_run(runs_dir)

    if not run_dir.exists():
        print(f"[ERROR] Run directory not found: {run_dir}")
        exit(1)

    # Visualize
    try:
        visualize_backtest(run_dir, show_equity=args.equity)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
