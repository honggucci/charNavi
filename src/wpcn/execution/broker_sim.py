from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

from wpcn.core.types import Theta, BacktestCosts, BacktestConfig
from wpcn.execution.cost import apply_costs
from wpcn.wyckoff.events import detect_spring_utad
from wpcn.engine.navigation import compute_navigation

def approx_pfill_det(df: pd.DataFrame, t_idx: int, entry_price: float, N_fill: int) -> float:
    end = min(len(df), t_idx + 1 + N_fill)
    window = df.iloc[t_idx+1:end]
    if window.empty:
        return 0.0
    hit = ((window["low"] <= entry_price) & (entry_price <= window["high"])).any()
    return 1.0 if hit else 0.0

def simulate(df: pd.DataFrame, theta: Theta, costs: BacktestCosts, cfg: BacktestConfig, mtf: List[str] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_index()
    mtf = mtf or []

    # navigation (page probs + Unknown gate)
    nav = compute_navigation(df, theta, cfg, mtf)

    signals = detect_spring_utad(df, theta)
    signals_df = pd.DataFrame([{
        "time": s.t_signal, "side": s.side, "event": s.event,
        "entry": s.entry_price, "stop": s.stop_price, "tp1": s.tp1, "tp2": s.tp2,
        **{f"meta_{k}": v for k,v in (s.meta or {}).items()}
    } for s in signals]).sort_values("time") if signals else pd.DataFrame(columns=["time"])

    cash = float(cfg.initial_equity)
    pos_qty = 0.0
    pos_side = 0  # +1 long, -1 short
    stop_price = np.nan
    tp1 = np.nan
    tp2 = np.nan
    entry_i = -1
    tp1_done = False

    pending_orders: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    sig_by_time: Dict[Any, List[Any]] = {}
    for s in signals:
        sig_by_time.setdefault(s.t_signal, []).append(s)

    for i, t in enumerate(df.index):
        o,h,l,c = df.loc[t, ["open","high","low","close"]].astype(float)

        # place new orders (activate next bar) if gate open
        if t in sig_by_time:
            gate_ok = bool(int(nav.loc[t, "trade_gate"])) if t in nav.index else True
            if gate_ok:
                for s in sig_by_time[t]:
                    pfill = approx_pfill_det(df, i, s.entry_price, theta.N_fill)
                    if pfill < theta.F_min:
                        continue
                    pending_orders.append({
                        "activate_i": i+1,
                        "expire_i": i+1+theta.N_fill,
                        "side": s.side,
                        "event": s.event,
                        "limit": float(s.entry_price),
                        "stop": float(s.stop_price),
                        "tp1": float(s.tp1),
                        "tp2": float(s.tp2),
                        "meta": s.meta or {}
                    })

        # fill pending if flat
        if pos_side == 0 and pending_orders:
            still = []
            for od in pending_orders:
                if i < od["activate_i"]:
                    still.append(od); continue
                if i > od["expire_i"]:
                    continue
                limit_px = od["limit"]
                if l <= limit_px <= h:
                    fill_px = apply_costs(limit_px, costs)
                    pos_side = +1 if od["side"] == "long" else -1
                    pos_qty = (cash / fill_px) * pos_side
                    cash = 0.0
                    stop_price = od["stop"]
                    tp1 = od["tp1"]
                    tp2 = od["tp2"]
                    entry_i = i
                    tp1_done = False
                    trades.append({"time": t, "type":"ENTRY", "side": od["side"], "price": fill_px, "event": od["event"], **od["meta"]})
                else:
                    still.append(od)
            pending_orders = still

        # manage position
        if pos_side != 0:
            # stop
            if (pos_side == 1 and l <= stop_price) or (pos_side == -1 and h >= stop_price):
                exit_px = apply_costs(stop_price, costs)
                cash = cash + pos_qty * exit_px
                trades.append({"time": t, "type":"STOP", "side":"long" if pos_side==1 else "short", "price": exit_px})
                pos_qty = 0.0; pos_side = 0; stop_price=np.nan; tp1=np.nan; tp2=np.nan; entry_i=-1; tp1_done=False
            else:
                # tp1
                if not tp1_done:
                    if (pos_side == 1 and h >= tp1) or (pos_side == -1 and l <= tp1):
                        exit_px = apply_costs(tp1, costs)
                        frac = cfg.tp1_frac
                        qty_exit = pos_qty * frac
                        cash = cash + qty_exit * exit_px
                        pos_qty = pos_qty * (1-frac)
                        tp1_done = True
                        trades.append({"time": t, "type":"TP1", "side":"long" if pos_side==1 else "short", "price": exit_px})
                # tp2
                if cfg.use_tp2 and pos_side != 0:
                    if (pos_side == 1 and h >= tp2) or (pos_side == -1 and l <= tp2):
                        exit_px = apply_costs(tp2, costs)
                        cash = cash + pos_qty * exit_px
                        trades.append({"time": t, "type":"TP2", "side":"long" if pos_side==1 else "short", "price": exit_px})
                        pos_qty = 0.0; pos_side = 0; stop_price=np.nan; tp1=np.nan; tp2=np.nan; entry_i=-1; tp1_done=False
                # max hold
                if pos_side != 0 and cfg.max_hold_bars is not None:
                    if (i - entry_i) >= cfg.max_hold_bars:
                        exit_px = apply_costs(c, costs)
                        cash = cash + pos_qty * exit_px
                        trades.append({"time": t, "type":"TIME_EXIT", "side":"long" if pos_side==1 else "short", "price": exit_px})
                        pos_qty = 0.0; pos_side = 0; stop_price=np.nan; tp1=np.nan; tp2=np.nan; entry_i=-1; tp1_done=False

        equity_val = cash + pos_qty * c
        equity_rows.append({"time": t, "equity": float(equity_val), "pos_qty": float(pos_qty), "pos_side": int(pos_side)})

    equity_df = pd.DataFrame(equity_rows).set_index("time")
    equity_df["ret"] = equity_df["equity"].pct_change().fillna(0.0)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["time","type","side","price"])
    return equity_df, trades_df, signals_df, nav
