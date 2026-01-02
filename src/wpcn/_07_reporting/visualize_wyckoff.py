"""
Wyckoff Backtest Visualizer - 와이코프 분석 포함

Usage:
    python visualize_wyckoff.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
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


def visualize_wyckoff(run_dir: Path):
    """Create comprehensive Wyckoff analysis chart"""

    print(f"[INFO] Loading data from: {run_dir.name}")

    # Load data
    candles = pd.read_parquet(run_dir / 'candles.parquet')
    trades = pd.read_parquet(run_dir / 'trades.parquet')
    equity = pd.read_parquet(run_dir / 'equity.parquet')
    nav = pd.read_parquet(run_dir / 'nav.parquet')

    print(f"  Candles: {len(candles):,}")
    print(f"  Trades: {len(trades):,}")
    print(f"  Navigation: {len(nav):,}")

    # Create figure with 4 subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(
            'Price & Wyckoff Analysis',
            'RSI',
            'Market Regime',
            'Equity Curve'
        )
    )

    # === 1. Candlestick chart ===
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

    # === 2. Wyckoff Box (High/Low/Mid) ===
    # Remove NaN values
    box_data = nav[['box_high', 'box_low', 'box_mid']].dropna()

    if len(box_data) > 0:
        # Box High (Resistance)
        fig.add_trace(
            go.Scatter(
                x=box_data.index,
                y=box_data['box_high'],
                mode='lines',
                line=dict(color='red', width=1, dash='dot'),
                name='Box High',
                opacity=0.6
            ),
            row=1, col=1
        )

        # Box Low (Support)
        fig.add_trace(
            go.Scatter(
                x=box_data.index,
                y=box_data['box_low'],
                mode='lines',
                line=dict(color='green', width=1, dash='dot'),
                name='Box Low',
                opacity=0.6
            ),
            row=1, col=1
        )

        # Box Mid
        fig.add_trace(
            go.Scatter(
                x=box_data.index,
                y=box_data['box_mid'],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                name='Box Mid',
                opacity=0.4
            ),
            row=1, col=1
        )

    # === 3. Entry signals ===
    entry_trades = trades[trades['type'] == 'ENTRY']
    if len(entry_trades) > 0:
        fig.add_trace(
            go.Scatter(
                x=entry_trades['time'],
                y=entry_trades['price'],
                mode='markers+text',
                marker=dict(
                    symbol='triangle-up',
                    size=20,
                    color='lime',
                    line=dict(color='darkgreen', width=2)
                ),
                text=['BUY' for _ in range(len(entry_trades))],
                textposition='bottom center',
                textfont=dict(size=10, color='lime'),
                name='Entry',
                hovertemplate='<b>ENTRY</b><br>Price: $%{y:.2f}<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )

    # === 3.5. Accumulation Buys (물타기) ===
    accum_buys = trades[trades['type'] == 'ACCUM_LONG']
    if len(accum_buys) > 0:
        fig.add_trace(
            go.Scatter(
                x=accum_buys['time'],
                y=accum_buys['price'],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=4,
                    color='yellow',
                    opacity=0.6
                ),
                name=f'Averaging ({len(accum_buys)})',
                hovertemplate='<b>AVERAGING</b><br>Price: $%{y:.2f}<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )

    # === 4. Exit signals ===
    exit_types = ['TP1', 'TP2', 'STOP', 'TIME_EXIT', 'ACCUM_TP']
    exit_trades = trades[trades['type'].isin(exit_types)]

    if len(exit_trades) > 0:
        tp_exits = exit_trades[exit_trades['type'].isin(['TP1', 'TP2', 'ACCUM_TP'])]
        sl_exits = exit_trades[exit_trades['type'] == 'STOP']
        time_exits = exit_trades[exit_trades['type'] == 'TIME_EXIT']

        if len(tp_exits) > 0:
            fig.add_trace(
                go.Scatter(
                    x=tp_exits['time'],
                    y=tp_exits['price'],
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', size=18, color='#26a69a',
                              line=dict(color='darkgreen', width=2)),
                    text=['TP' for _ in range(len(tp_exits))],
                    textposition='top center',
                    textfont=dict(size=9, color='#26a69a'),
                    name='Take Profit',
                    hovertemplate='<b>TP</b><br>$%{y:.2f}<br>%{x}<extra></extra>'
                ),
                row=1, col=1
            )

        if len(sl_exits) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sl_exits['time'],
                    y=sl_exits['price'],
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', size=18, color='#ef5350',
                              line=dict(color='darkred', width=2)),
                    text=['SL' for _ in range(len(sl_exits))],
                    textposition='top center',
                    textfont=dict(size=9, color='#ef5350'),
                    name='Stop Loss',
                    hovertemplate='<b>SL</b><br>$%{y:.2f}<br>%{x}<extra></extra>'
                ),
                row=1, col=1
            )

        if len(time_exits) > 0:
            fig.add_trace(
                go.Scatter(
                    x=time_exits['time'],
                    y=time_exits['price'],
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', size=18, color='#ffeb3b',
                              line=dict(color='orange', width=2)),
                    text=['TIME' for _ in range(len(time_exits))],
                    textposition='top center',
                    textfont=dict(size=9, color='#ffeb3b'),
                    name='Time Exit',
                    hovertemplate='<b>TIME</b><br>$%{y:.2f}<br>%{x}<extra></extra>'
                ),
                row=1, col=1
            )

    # === 5. Spring/UTAD Events (from nav) ===
    # Detect high probability spring/utad
    spring_events = nav[(nav['p_spring'] > 0.5) & (nav['edge_event'].notna())]
    utad_events = nav[(nav['p_utad'] > 0.5) & (nav['edge_event'].notna())]

    if len(spring_events) > 0:
        # Get corresponding prices from candles
        spring_prices = candles.loc[spring_events.index, 'low']
        fig.add_trace(
            go.Scatter(
                x=spring_events.index,
                y=spring_prices,
                mode='markers',
                marker=dict(symbol='star', size=15, color='cyan',
                          line=dict(color='blue', width=2)),
                name='Spring Event',
                hovertemplate='<b>SPRING</b><br>$%{y:.2f}<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )

    if len(utad_events) > 0:
        utad_prices = candles.loc[utad_events.index, 'high']
        fig.add_trace(
            go.Scatter(
                x=utad_events.index,
                y=utad_prices,
                mode='markers',
                marker=dict(symbol='star', size=15, color='orange',
                          line=dict(color='red', width=2)),
                name='UTAD Event',
                hovertemplate='<b>UTAD</b><br>$%{y:.2f}<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )

    # === 6. RSI ===
    fig.add_trace(
        go.Scatter(
            x=nav.index,
            y=nav['rsi'],
            mode='lines',
            line=dict(color='cyan', width=1),
            name='RSI'
        ),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

    # === 7. Market Regime ===
    regime_colors = {
        'CHAOS': 'red',
        'RANGE': 'orange',
        'TREND_UP': 'green',
        'TREND_DOWN': 'blue'
    }

    # Create regime bands
    for regime, color in regime_colors.items():
        regime_mask = nav['regime'] == regime
        if regime_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=nav.index,
                    y=regime_mask.astype(int),
                    mode='lines',
                    fill='tozeroy',
                    line=dict(width=0),
                    fillcolor=color,
                    opacity=0.3,
                    name=regime,
                    showlegend=True,
                    hovertemplate=f'<b>{regime}</b><extra></extra>'
                ),
                row=3, col=1
            )

    # === 8. Equity curve ===
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity['equity'],
            mode='lines',
            line=dict(color='purple', width=2),
            name='Equity',
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)',
            hovertemplate='Equity: $%{y:.2f}<extra></extra>'
        ),
        row=4, col=1
    )

    # Calculate performance metrics for title
    initial = equity['equity'].iloc[0]
    final = equity['equity'].iloc[-1]
    total_return = (final / initial - 1) * 100

    profitable_exits = ((exit_trades.get('pnl', 0) > 0) |
                       (exit_trades.get('pnl_pct', 0) > 0)).sum() if len(exit_trades) > 0 else 0
    win_rate = profitable_exits / len(exit_trades) * 100 if len(exit_trades) > 0 else 0

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Wyckoff Backtest | Return: {total_return:+.2f}% | Win Rate: {win_rate:.1f}% | Trades: {len(exit_trades)}',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#667eea')
        ),
        height=1400,
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update axes
    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Regime", showticklabels=False, row=3, col=1)
    fig.update_yaxes(title_text="Equity (USD)", row=4, col=1)

    # Print summary
    print(f"\n[WYCKOFF ANALYSIS]")
    print(f"  Period: {candles.index[0]} -> {candles.index[-1]}")
    print(f"  Spring Events: {len(spring_events)}")
    print(f"  UTAD Events: {len(utad_events)}")
    print(f"  Entry signals: {len(entry_trades)}")
    print(f"  Exit signals: {len(exit_trades)}")

    # Regime distribution
    regime_dist = nav['regime'].value_counts()
    print(f"\n[REGIME DISTRIBUTION]")
    for regime, count in regime_dist.items():
        pct = count / len(nav) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")

    if len(equity) > 0:
        initial = equity['equity'].iloc[0]
        final = equity['equity'].iloc[-1]
        total_return = (final / initial - 1) * 100
        print(f"\n[PERFORMANCE]")
        print(f"  Initial Capital: ${initial:,.2f}")
        print(f"  Final Equity: ${final:,.2f}")
        print(f"  Total Return: {total_return:+.2f}%")

    if len(exit_trades) > 0:
        profitable = ((exit_trades.get('pnl', 0) > 0) |
                     (exit_trades.get('pnl_pct', 0) > 0)).sum()
        win_rate = profitable / len(exit_trades) * 100
        print(f"  Win Rate: {win_rate:.1f}%")

    # Save to HTML
    output_file = run_dir / 'wyckoff_analysis.html'
    fig.write_html(str(output_file))

    print(f"\n[SUCCESS] Chart saved to: {output_file}")
    print(f"[INFO] Opening in browser...")

    # Open in browser
    webbrowser.open(f'file:///{output_file.absolute()}')

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Visualize Wyckoff backtest analysis')
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
        visualize_wyckoff(run_dir)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
