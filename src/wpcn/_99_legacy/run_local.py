from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from wpcn.core.types import BacktestCosts, BacktestConfig, RunConfig
from wpcn.data.ccxt_fetch import fetch_ohlcv_to_parquet
from wpcn.data.loaders import load_parquet
from wpcn.engine.backtest import BacktestEngine, DEFAULT_THETA
from wpcn.tuning.theta_space import ThetaSpace
from wpcn.tuning.walk_forward import tune

def load_defaults() -> dict:
    with open(REPO_ROOT / "configs" / "run.default.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    d = load_defaults()

    ap = argparse.ArgumentParser("WPCN local backtest (no web)")
    ap.add_argument("--exchange", default=d.get("exchange_id","binanceusdm"))
    ap.add_argument("--symbol", default=d.get("symbol","BTC/USDT"))
    ap.add_argument("--timeframe", default=d.get("timeframe","5m"))
    ap.add_argument("--days", type=int, default=int(d.get("days",14)))
    ap.add_argument("--out", default=None, help="optional parquet path")
    ap.add_argument("--force-fetch", action="store_true")

    ap.add_argument("--mtf", default=",".join(d.get("mtf", [])), help="comma-separated: 15m,1h,4h,1d")

    ap.add_argument("--fee-bps", type=float, default=float(d.get("fee_bps",2.0)))
    ap.add_argument("--slippage-bps", type=float, default=float(d.get("slippage_bps",3.0)))

    ap.add_argument("--initial-equity", type=float, default=float(d.get("initial_equity",1.0)))
    ap.add_argument("--max-hold-bars", type=int, default=int(d.get("max_hold_bars",200)))
    ap.add_argument("--tp1-frac", type=float, default=float(d.get("tp1_frac",0.5)))
    ap.add_argument("--use-tp2", type=int, default=int(d.get("use_tp2",1)))

    ap.add_argument("--conf-min", type=float, default=float(d.get("conf_min",0.55)))
    ap.add_argument("--edge-min", type=float, default=float(d.get("edge_min",0.75)))
    ap.add_argument("--confirm-bars", type=int, default=int(d.get("confirm_bars",1)))
    ap.add_argument("--chaos-ret-atr", type=float, default=float(d.get("chaos_ret_atr",3.0)))
    ap.add_argument("--adx-len", type=int, default=int(d.get("adx_len",14)))
    ap.add_argument("--adx-trend", type=float, default=float(d.get("adx_trend",20.0)))

    ap.add_argument("--use-tuning", type=int, default=0)

    args = ap.parse_args()

    safe_symbol = args.symbol.replace("/", "-").replace(":", "_")
    data_path = Path(args.out) if args.out else (REPO_ROOT / "data" / "raw" / f"{args.exchange}_{safe_symbol}_{args.timeframe}.parquet")
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if args.force_fetch or (not data_path.exists()):
        fetch_ohlcv_to_parquet(
            exchange_id=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days,
            out_path=str(data_path),
            api_key=None,
            secret=None,
        )

    df = load_parquet(str(data_path))
    mtf = [x.strip() for x in (args.mtf or "").split(",") if x.strip()]

    costs = BacktestCosts(fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    bt = BacktestConfig(
        initial_equity=args.initial_equity,
        max_hold_bars=args.max_hold_bars,
        tp1_frac=args.tp1_frac,
        use_tp2=bool(args.use_tp2),

        conf_min=args.conf_min,
        edge_min=args.edge_min,
        confirm_bars=args.confirm_bars,
        chaos_ret_atr=args.chaos_ret_atr,
        adx_len=args.adx_len,
        adx_trend=args.adx_trend,
    )
    cfg = RunConfig(
        exchange_id=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        costs=costs,
        bt=bt,
        use_tuning=bool(args.use_tuning),
        wf=d.get("wf", {}),
        mtf=mtf,
        data_path=str(data_path),
    )

    theta = DEFAULT_THETA
    if args.use_tuning:
        space = ThetaSpace.from_yaml(str(REPO_ROOT / "configs" / "theta.space.yml"), seed=42)
        _, best_theta = tune(df, cfg, space)
        theta = best_theta
        print("Best theta:", best_theta)

    engine = BacktestEngine(runs_root=str(REPO_ROOT / "runs"))
    run_id = engine.run(df, cfg, theta=theta)
    print("Run complete:", run_id)
    print("Outputs under:", (REPO_ROOT / "runs" / run_id).resolve())

if __name__ == "__main__":
    main()
