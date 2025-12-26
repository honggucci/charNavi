# WPCN Backtester (CLI-only)
This package runs the Wyckoff Probabilistic Chart Navigation (WPCN) MVP backtest locally (no Flask).

## Install (recommended to avoid WinError 5 on Windows)
Use a fresh env (conda or venv). Example with conda:

```bash
conda create -n wpcn python=3.11 -y
conda activate wpcn
pip install -r requirements-cli.txt
```

## Run
Fetch + backtest (auto-download OHLCV to Parquet if missing):
```bash
python run_local.py --exchange binanceusdm --symbol BTC/USDT --timeframe 5m --days 14
```

Or fetch directly into memory and backtest:
```bash
python run_fetch_and_backtest.py --exchange binanceusdm --symbol BTC/USDT --timeframe 1h --days 7
```

Manual trading costs:
```bash
python run_local.py --fee-bps 2 --slippage-bps 3
```

Multi-timeframe (fractal) confirmation:
```bash
python run_local.py --mtf "15m,1h,4h,1d"
```

## Outputs
Saved under `runs/<run_id>/`:
- `candles.parquet`
- `equity.parquet`
- `trades.parquet`
- `signals.parquet`
- `nav.parquet` (page probabilities + gate + **Fibonacci target candidates with probabilities**)
- `reports/key_metrics.json`

## Features
- Wyckoff state machine (Accumulation/Distribution phases)
- Spring/UTAD event detection
- Multi-timeframe confirmation
- **Fibonacci target engine**: Generates support/resistance levels with probability scores based on RSI, Stoch RSI, and Wyckoff context
