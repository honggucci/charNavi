from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

from wpcn.core.types import Theta, RunConfig
from wpcn.execution.broker_sim import simulate
from wpcn.metrics.performance import summarize_performance
from wpcn.tuning.theta_space import ThetaSpace

def walk_forward_splits(n: int, train: int, test: int, embargo: int) -> List[Tuple[slice, slice]]:
    splits = []
    start = 0
    while True:
        tr0 = start
        tr1 = tr0 + train
        te0 = tr1 + embargo
        te1 = te0 + test
        if te1 > n:
            break
        splits.append((slice(tr0, tr1), slice(te0, te1)))
        start = te0
    return splits

def objective(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> float:
    # conservative: mean ret - |maxdd| - turnover penalty
    perf = summarize_performance(equity_df, trades_df)
    mean_ret = float(equity_df["ret"].mean())
    mdd = abs(perf["max_drawdown_pct"]) / 100.0
    tc = perf["trade_entries"]
    return mean_ret - 1.0*mdd - 0.0005*tc

def tune(df: pd.DataFrame, cfg: RunConfig, theta_space: ThetaSpace) -> Tuple[pd.DataFrame, Theta]:
    wf = cfg.wf
    splits = walk_forward_splits(len(df), int(wf["train_bars"]), int(wf["test_bars"]), int(wf["embargo_bars"]))
    rows = []
    best_theta: Optional[Theta] = None
    best_obj = -1e9

    for fold_id, (tr_sl, te_sl) in enumerate(splits):
        train_df = df.iloc[tr_sl]
        test_df = df.iloc[te_sl]

        fold_best_theta = None
        fold_best_obj = -1e9

        for _ in range(int(wf["n_candidates"])):
            th = theta_space.sample()
            eq_tr, tr_trades, _ = simulate(train_df, th, cfg.costs, cfg.bt)
            obj = objective(eq_tr, tr_trades)
            if obj > fold_best_obj:
                fold_best_obj = obj
                fold_best_theta = th

        assert fold_best_theta is not None
        eq_te, te_trades, _ = simulate(test_df, fold_best_theta, cfg.costs, cfg.bt)
        obj_te = objective(eq_te, te_trades)
        rows.append({
            "fold_id": fold_id,
            "objective_train": fold_best_obj,
            "objective_test": obj_te,
            **asdict(fold_best_theta),
        })

        if obj_te > best_obj:
            best_obj = obj_te
            best_theta = fold_best_theta

    wf_table = pd.DataFrame(rows)
    assert best_theta is not None
    return wf_table, best_theta
