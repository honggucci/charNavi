from __future__ import annotations
import argparse
from pathlib import Path
import yaml

from wpcn.core.types import BacktestCosts, BacktestConfig, RunConfig
from wpcn.data.loaders import load_parquet
from wpcn.data.ccxt_fetch import fetch_ohlcv_to_parquet
from wpcn.engine.backtest import BacktestEngine, DEFAULT_THETA
from wpcn.tuning.theta_space import ThetaSpace
from wpcn.tuning.walk_forward import tune

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def cmd_fetch(args):
    out = args.out
    if out is None:
        safe_symbol = args.symbol.replace("/", "-").replace(":", "_")
        out = f"data/raw/{args.exchange}_{safe_symbol}_{args.timeframe}.parquet"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fetch_ohlcv_to_parquet(args.exchange, args.symbol, args.timeframe, args.days, out, limit=args.limit)
    print(f"Saved: {out}")

def cmd_backtest(args):
    df = load_parquet(args.data)
    defaults = _load_yaml("configs/run.default.yml")

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
    mtf = [x.strip() for x in (args.mtf or ",".join(defaults.get("mtf", []))).split(",") if x.strip()]

    cfg = RunConfig(
        exchange_id=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=int(args.days),
        costs=costs,
        bt=bt,
        use_tuning=bool(args.use_tuning),
        wf=defaults["wf"],
        mtf=mtf,
        data_path=args.data
    )
    engine = BacktestEngine(runs_root="runs")
    theta = DEFAULT_THETA

    if args.use_tuning:
        space = ThetaSpace.from_yaml("configs/theta.space.yml", seed=42)
        wf_table, best_theta = tune(df, cfg, space)
        print("Best theta:", best_theta)
        theta = best_theta

    run_id = engine.run(df, cfg, theta=theta)
    print("Run:", run_id)

def build_parser():
    p = argparse.ArgumentParser(prog="wpcn")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch futures OHLCV via ccxt and save as parquet")
    p_fetch.add_argument("--exchange", default="binanceusdm")
    p_fetch.add_argument("--symbol", default="BTC/USDT")
    p_fetch.add_argument("--timeframe", default="5m")
    p_fetch.add_argument("--days", type=int, default=14)
    p_fetch.add_argument("--limit", type=int, default=1500)
    p_fetch.add_argument("--out", default=None)
    p_fetch.set_defaults(func=cmd_fetch)

    p_bt = sub.add_parser("backtest", help="Run backtest on a parquet OHLCV file")
    p_bt.add_argument("--data", required=True)
    p_bt.add_argument("--exchange", default="binanceusdm")
    p_bt.add_argument("--symbol", default="BTC/USDT")
    p_bt.add_argument("--timeframe", default="5m")
    p_bt.add_argument("--days", type=int, default=14)

    p_bt.add_argument("--mtf", default=None, help="comma-separated, e.g. 15m,1h,4h,1d")

    p_bt.add_argument("--fee-bps", dest="fee_bps", type=float, default=2.0)
    p_bt.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=3.0)
    p_bt.add_argument("--initial-equity", type=float, default=1.0)
    p_bt.add_argument("--max-hold-bars", type=int, default=200)
    p_bt.add_argument("--tp1-frac", type=float, default=0.5)
    p_bt.add_argument("--use-tp2", type=int, default=1)

    p_bt.add_argument("--conf-min", dest="conf_min", type=float, default=0.55)
    p_bt.add_argument("--edge-min", dest="edge_min", type=float, default=0.75)
    p_bt.add_argument("--confirm-bars", dest="confirm_bars", type=int, default=1)
    p_bt.add_argument("--chaos-ret-atr", dest="chaos_ret_atr", type=float, default=3.0)
    p_bt.add_argument("--adx-len", dest="adx_len", type=int, default=14)
    p_bt.add_argument("--adx-trend", dest="adx_trend", type=float, default=20.0)

    p_bt.add_argument("--use-tuning", type=int, default=0)
    p_bt.set_defaults(func=cmd_backtest)

    return p

def main():
    p = build_parser()
    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
