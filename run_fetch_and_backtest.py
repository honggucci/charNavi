import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(str(ROOT))

import argparse
from wpcn.core.types import BacktestCosts, BacktestConfig, RunConfig
from wpcn.data.ccxt_fetch import fetch_ohlcv_df
from wpcn.engine.backtest import BacktestEngine, DEFAULT_THETA


def build_parser():
    p = argparse.ArgumentParser(prog="wpcn-fetch-backtest")
    p.add_argument("--exchange", default="binanceusdm")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--limit", type=int, default=1500)

    p.add_argument("--fee-bps", dest="fee_bps", type=float, default=2.0)
    p.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=3.0)
    p.add_argument("--initial-equity", type=float, default=1.0)
    p.add_argument("--max-hold-bars", type=int, default=200)
    p.add_argument("--tp1-frac", type=float, default=0.5)
    p.add_argument("--use-tp2", type=int, default=1)

    p.add_argument("--conf-min", dest="conf_min", type=float, default=0.55)
    p.add_argument("--edge-min", dest="edge_min", type=float, default=0.75)
    p.add_argument("--confirm-bars", dest="confirm_bars", type=int, default=1)
    p.add_argument("--chaos-ret-atr", dest="chaos_ret_atr", type=float, default=3.0)
    p.add_argument("--adx-len", dest="adx_len", type=int, default=14)
    p.add_argument("--adx-trend", dest="adx_trend", type=float, default=20.0)

    p.add_argument("--use-tuning", type=int, default=0)
    return p


def main():
    p = build_parser()
    args = p.parse_args()

    print(f"Fetching {args.exchange} {args.symbol} {args.timeframe} last {args.days} days...")
    df = fetch_ohlcv_df(args.exchange, args.symbol, args.timeframe, args.days, limit=args.limit)
    print("Fetched bars:", len(df))

    costs = BacktestCosts(fee_bps=float(args.fee_bps), slippage_bps=float(args.slippage_bps))
    bt = BacktestConfig(
        initial_equity=float(args.initial_equity),
        max_hold_bars=int(args.max_hold_bars),
        tp1_frac=float(args.tp1_frac),
        use_tp2=bool(args.use_tp2),
        conf_min=float(args.conf_min),
        edge_min=float(args.edge_min),
        confirm_bars=int(args.confirm_bars),
        chaos_ret_atr=float(args.chaos_ret_atr),
        adx_len=int(args.adx_len),
        adx_trend=float(args.adx_trend),
    )

    cfg = RunConfig(
        exchange_id=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=int(args.days),
        costs=costs,
        bt=bt,
        use_tuning=bool(args.use_tuning),
        wf=None,
        mtf=[],
        data_path=None,
    )

    engine = BacktestEngine(runs_root="runs")
    theta = DEFAULT_THETA

    run_id = engine.run(df, cfg, theta=theta)
    print("Run complete:", run_id)


if __name__ == "__main__":
    main()
