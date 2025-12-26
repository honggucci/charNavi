from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())

def summarize_performance(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
    eq = equity_df["equity"]
    rets = equity_df["ret"]
    mdd = _max_drawdown(eq)
    sharpe = float((rets.mean() / (rets.std(ddof=0) + 1e-12)) * np.sqrt(365))  # rough
    entries = int((trades_df["type"] == "ENTRY").sum()) if len(trades_df) else 0
    stops = int((trades_df["type"] == "STOP").sum()) if len(trades_df) else 0
    tps = int(((trades_df["type"] == "TP1") | (trades_df["type"] == "TP2")).sum()) if len(trades_df) else 0

    return {
        "final_equity": float(eq.iloc[-1]),
        "total_return_pct": float((eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0),
        "max_drawdown_pct": float(mdd * 100.0),
        "sharpe_proxy": sharpe,
        "trade_entries": entries,
        "tp_events": tps,
        "stop_events": stops,
    }
