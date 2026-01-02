"""
Backtest Results Visualizer using Plotly (Interactive HTML)

Usage:
    python visualize_backtest_plotly.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser


def get_latest_run(runs_dir: Path) -> Path:
    """Get the most recent backtest run directory"""
    runs = list(runs_dir.glob('*'))
    if not runs:
        raise FileNotFoundError(f"No runs found in {runs_dir}")
    latest = sorted(runs, reverse=True)[0]
    return latest


def visualize_backtest_plotly(run_dir: Path):
    """Create interactive Plotly chart"""

    print(f"[INFO] Loading data from: {run_dir.name}")

    # Load data
    candles = pd.read_parquet(run_dir / 'candles.parquet')
    trades = pd.read_parquet(run_dir / 'trades.parquet')
    equity = pd.read_parquet(run_dir / 'equity.parquet')

    print(f"  Candles: {len(candles):,}")
    print(f"  Trades: {len(trades):,}")

    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('BTC-USDT Price & Trades', 'Equity Curve')
    )

    # === Candlestick chart ===
    fig.add_trace(
        go.Candlestick(
            x=candles.index,
            open=candles['open'],
            high=candles['high'],
            low=candles['low'],
            close=candles['close'],
            name='BTC-USDT',
            showlegend=False
        ),
        row=1, col=1
    )

    # === Entry signals ===
    entry_trades = trades[trades['type'] == 'ENTRY']
    if len(entry_trades) > 0:
        fig.add_trace(
            go.Scatter(
                x=entry_trades['time'],
                y=entry_trades['price'],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='green',
                    line=dict(color='darkgreen', width=2)
                ),
                name='Entry',
                text=[f"BUY @ ${p:.0f}" for p in entry_trades['price']],
                hovertemplate='<b>ENTRY</b><br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
            ),
            row=1, col=1
        )

    # === Exit signals ===
    exit_types = ['TP1', 'TP2', 'STOP', 'TIME_EXIT', 'ACCUM_TP']
    exit_trades = trades[trades['type'].isin(exit_types)]

    if len(exit_trades) > 0:
        # Group by exit type
        tp_exits = exit_trades[exit_trades['type'].isin(['TP1', 'TP2', 'ACCUM_TP'])]
        sl_exits = exit_trades[exit_trades['type'] == 'STOP']
        time_exits = exit_trades[exit_trades['type'] == 'TIME_EXIT']

        # TP exits (green)
        if len(tp_exits) > 0:
            fig.add_trace(
                go.Scatter(
                    x=tp_exits['time'],
                    y=tp_exits['price'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='#26a69a',
                        line=dict(color='darkgreen', width=2)
                    ),
                    name='Take Profit',
                    text=[f"TP @ ${p:.0f}" for p in tp_exits['price']],
                    hovertemplate='<b>TAKE PROFIT</b><br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )

        # SL exits (red)
        if len(sl_exits) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sl_exits['time'],
                    y=sl_exits['price'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='#ef5350',
                        line=dict(color='darkred', width=2)
                    ),
                    name='Stop Loss',
                    text=[f"SL @ ${p:.0f}" for p in sl_exits['price']],
                    hovertemplate='<b>STOP LOSS</b><br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )

        # Time exits (yellow)
        if len(time_exits) > 0:
            fig.add_trace(
                go.Scatter(
                    x=time_exits['time'],
                    y=time_exits['price'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='#ffeb3b',
                        line=dict(color='orange', width=2)
                    ),
                    name='Time Exit',
                    text=[f"TIME @ ${p:.0f}" for p in time_exits['price']],
                    hovertemplate='<b>TIME EXIT</b><br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )

    # === Equity curve ===
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity['equity'],
            mode='lines',
            line=dict(color='purple', width=2),
            name='Equity',
            hovertemplate='Equity: $%{y:.2f}<br>Time: %{x}<extra></extra>'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Wyckoff Backtest Results - {run_dir.name}',
            x=0.5,
            xanchor='center'
        ),
        height=1000,
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )

    # Update axes
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Equity (USD)", row=2, col=1)

    # Print summary
    print(f"\n[SUMMARY]")
    print(f"  Period: {candles.index[0]} -> {candles.index[-1]}")
    print(f"  Entry signals: {len(entry_trades)}")
    print(f"  Exit signals: {len(exit_trades)}")

    if len(equity) > 0:
        initial = equity['equity'].iloc[0]
        final = equity['equity'].iloc[-1]
        total_return = (final / initial - 1) * 100
        print(f"  Initial Capital: ${initial:,.2f}")
        print(f"  Final Equity: ${final:,.2f}")
        print(f"  Total Return: {total_return:+.2f}%")

    if len(exit_trades) > 0:
        profitable = ((exit_trades.get('pnl', 0) > 0) |
                     (exit_trades.get('pnl_pct', 0) > 0)).sum()
        win_rate = profitable / len(exit_trades) * 100
        print(f"  Win Rate: {win_rate:.1f}%")

    # Save to HTML
    output_file = run_dir / 'backtest_chart.html'
    fig.write_html(str(output_file))

    print(f"\n[SUCCESS] Chart saved to: {output_file}")
    print(f"[INFO] Opening in browser...")

    # Open in browser
    webbrowser.open(f'file:///{output_file.absolute()}')

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Visualize backtest with Plotly')
    parser.add_argument('--run', type=str, default=None,
                       help='Path to run directory')
    args = parser.parse_args()

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

    try:
        visualize_backtest_plotly(run_dir)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
