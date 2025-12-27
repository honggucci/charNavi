from __future__ import annotations
from dataclasses import asdict
from typing import Optional

import pandas as pd

from wpcn.core.types import RunConfig, Theta
from wpcn.execution.broker_sim import simulate
from wpcn.engine.run_manager import make_run_id, ensure_run_dir, dump_resolved_config, save_parquet as save_pq, save_json, write_review_packet
from wpcn.metrics.performance import summarize_performance
from wpcn.reporting.review_packet import build_review_prompt
from wpcn.reporting.charts import plot_backtest_results

DEFAULT_THETA = Theta(
    pivot_lr=4, box_L=96, m_freeze=32, atr_len=14,
    x_atr=0.35, m_bw=0.10, N_reclaim=3,
    N_fill=5, F_min=0.70
)

class BacktestEngine:
    def __init__(self, runs_root: str = "runs"):
        self.runs_root = runs_root

    def run(self, df: pd.DataFrame, cfg: RunConfig, theta: Optional[Theta] = None, scalping_mode: bool = False, use_phase_accumulation: bool = True) -> str:
        theta = theta or DEFAULT_THETA
        run_id = make_run_id(cfg.exchange_id, cfg.symbol, cfg.timeframe)
        run_dir = ensure_run_dir(self.runs_root, run_id)

        resolved = {
            "exchange_id": cfg.exchange_id, "symbol": cfg.symbol, "timeframe": cfg.timeframe, "days": cfg.days,
            "costs": asdict(cfg.costs),
            "bt": asdict(cfg.bt),
            "use_tuning": cfg.use_tuning,
            "wf": cfg.wf,
            "mtf": cfg.mtf,
            "theta": asdict(theta),
            "data_path": cfg.data_path,
            "scalping_mode": scalping_mode,
            "use_phase_accumulation": use_phase_accumulation,
        }
        dump_resolved_config(run_dir, resolved)

        equity_df, trades_df, signals_df, nav_df = simulate(
            df, theta, cfg.costs, cfg.bt,
            mtf=cfg.mtf,
            scalping_mode=scalping_mode,
            use_phase_accumulation=use_phase_accumulation
        )

        save_pq(df, run_dir / "candles.parquet")
        save_pq(equity_df, run_dir / "equity.parquet")
        save_pq(trades_df, run_dir / "trades.parquet")
        save_pq(signals_df, run_dir / "signals.parquet")
        save_pq(nav_df, run_dir / "nav.parquet")

        # quick UI snapshot (last bar)
        last = nav_df.iloc[-1].to_dict()
        save_json(last, run_dir / "reports" / "nav_last.json")

        metrics = summarize_performance(equity_df, trades_df)
        save_json(metrics, run_dir / "reports" / "key_metrics.json")

        summary_md = f"""# WPCN Backtest Summary
- Run: {run_id}
- Exchange: {cfg.exchange_id}
- Symbol: {cfg.symbol}
- Timeframe: {cfg.timeframe}
- Bars: {len(df)}

## Key Metrics
```json
{metrics}
```
"""
        prompt = build_review_prompt()
        write_review_packet(run_dir, summary_md, metrics, prompt)

        # Generate charts
        plot_backtest_results(run_dir, df, equity_df, trades_df, nav_df)

        return run_id
