"""
Wyckoff Backtest Dashboard - Flask Web App
"""
from flask import Flask, render_template, jsonify, request, send_file
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

app = Flask(__name__)

# Configuration
RUNS_DIR = Path('runs')


def get_all_runs():
    """Get list of all backtest runs"""
    if not RUNS_DIR.exists():
        return []

    runs = []
    for run_dir in sorted(RUNS_DIR.glob('*'), reverse=True):
        if not run_dir.is_dir():
            continue

        # Parse run directory name
        # Format: 2025-12-31_19-31-06_binance_BTC-USDT_15m
        parts = run_dir.name.split('_')

        try:
            # Load basic info
            equity_path = run_dir / 'equity.parquet'
            if equity_path.exists():
                equity = pd.read_parquet(equity_path)
                initial = equity['equity'].iloc[0]
                final = equity['equity'].iloc[-1]
                returns = (final / initial - 1) * 100
            else:
                returns = 0

            trades_path = run_dir / 'trades.parquet'
            num_trades = len(pd.read_parquet(trades_path)) if trades_path.exists() else 0

            runs.append({
                'name': run_dir.name,
                'date': parts[0] if len(parts) > 0 else '',
                'time': parts[1] if len(parts) > 1 else '',
                'symbol': parts[3] if len(parts) > 3 else '',
                'timeframe': parts[4] if len(parts) > 4 else '',
                'returns': f"{returns:+.2f}%",
                'returns_value': returns,
                'num_trades': num_trades
            })
        except Exception as e:
            print(f"Error loading run {run_dir.name}: {e}")
            continue

    return runs


def load_run_data(run_name):
    """Load complete run data"""
    run_dir = RUNS_DIR / run_name

    if not run_dir.exists():
        return None

    data = {}

    # Load candles
    candles_path = run_dir / 'candles.parquet'
    if candles_path.exists():
        candles = pd.read_parquet(candles_path)
        data['candles'] = candles.reset_index().to_dict(orient='records')

    # Load trades
    trades_path = run_dir / 'trades.parquet'
    if trades_path.exists():
        trades = pd.read_parquet(trades_path)
        data['trades'] = trades.to_dict(orient='records')

    # Load equity
    equity_path = run_dir / 'equity.parquet'
    if equity_path.exists():
        equity = pd.read_parquet(equity_path)
        data['equity'] = equity.reset_index().to_dict(orient='records')

    # Load navigation
    nav_path = run_dir / 'nav.parquet'
    if nav_path.exists():
        nav = pd.read_parquet(nav_path)
        data['nav'] = nav.reset_index().to_dict(orient='records')

    return data


@app.route('/')
def index():
    """Main dashboard page"""
    runs = get_all_runs()
    return render_template('index.html', runs=runs)


@app.route('/api/runs')
def api_runs():
    """API endpoint for runs list"""
    runs = get_all_runs()
    return jsonify(runs)


@app.route('/api/run/<run_name>')
def api_run_data(run_name):
    """API endpoint for specific run data"""
    data = load_run_data(run_name)
    if data is None:
        return jsonify({'error': 'Run not found'}), 404
    return jsonify(data)


@app.route('/run/<run_name>')
def view_run(run_name):
    """View specific run"""
    run_dir = RUNS_DIR / run_name
    if not run_dir.exists():
        return "Run not found", 404

    return render_template('run_detail.html', run_name=run_name)


@app.route('/api/chart/<run_name>')
def api_chart_data(run_name):
    """Get chart data for a specific run"""
    run_dir = RUNS_DIR / run_name

    if not run_dir.exists():
        return jsonify({'error': 'Run not found'}), 404

    # Load data
    candles = pd.read_parquet(run_dir / 'candles.parquet')
    trades = pd.read_parquet(run_dir / 'trades.parquet')
    nav = pd.read_parquet(run_dir / 'nav.parquet')
    equity = pd.read_parquet(run_dir / 'equity.parquet')

    # Prepare chart data
    chart_data = {
        'candles': {
            'time': candles.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'open': candles['open'].tolist(),
            'high': candles['high'].tolist(),
            'low': candles['low'].tolist(),
            'close': candles['close'].tolist(),
            'volume': candles['volume'].tolist()
        },
        'trades': {
            'entry': [],
            'tp': [],
            'sl': [],
            'time_exit': []
        },
        'wyckoff': {
            'box_high': nav['box_high'].ffill().tolist(),
            'box_low': nav['box_low'].ffill().tolist(),
            'box_mid': nav['box_mid'].ffill().tolist()
        },
        'equity': {
            'time': equity.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'value': equity['equity'].tolist()
        },
        'rsi': {
            'time': nav.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'value': nav['rsi'].tolist()
        }
    }

    # Parse trades
    entry_trades = trades[trades['type'] == 'ENTRY']
    for _, trade in entry_trades.iterrows():
        chart_data['trades']['entry'].append({
            'time': trade['time'].strftime('%Y-%m-%d %H:%M:%S'),
            'price': float(trade['price'])
        })

    exit_types = {
        'tp': ['TP1', 'TP2', 'ACCUM_TP'],
        'sl': ['STOP'],
        'time_exit': ['TIME_EXIT']
    }

    for exit_type, types in exit_types.items():
        exit_trades = trades[trades['type'].isin(types)]
        for _, trade in exit_trades.iterrows():
            chart_data['trades'][exit_type].append({
                'time': trade['time'].strftime('%Y-%m-%d %H:%M:%S'),
                'price': float(trade['price'])
            })

    return jsonify(chart_data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
