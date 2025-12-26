from __future__ import annotations
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from wpcn.core.types import Theta, EventSignal
from wpcn.features.indicators import atr
from wpcn.wyckoff.box import box_engine_freeze

def detect_spring_utad(df: pd.DataFrame, theta: Theta) -> List[EventSignal]:
    _atr = atr(df, theta.atr_len)
    box = box_engine_freeze(df, theta)
    df2 = df.join(_atr.rename("atr")).join(box)

    signals: List[EventSignal] = []
    pending: Optional[Dict[str, Any]] = None

    for i in range(len(df2)):
        row = df2.iloc[i]
        if any(np.isnan(row[x]) for x in ["atr","box_low","box_high","box_width"]):
            continue

        bl, bh, bw = float(row["box_low"]), float(row["box_high"]), float(row["box_width"])
        at = float(row["atr"])
        close, low, high = float(row["close"]), float(row["low"]), float(row["high"])

        # expire pending break
        if pending is not None and i > pending["break_i"] + theta.N_reclaim:
            pending = None

        # new break
        if pending is None:
            if low < bl - theta.x_atr * at:
                pending = {"type":"spring", "break_i": i, "bl": bl, "bh": bh, "bw": bw, "atr": at}
                continue
            if high > bh + theta.x_atr * at:
                pending = {"type":"utad", "break_i": i, "bl": bl, "bh": bh, "bw": bw, "atr": at}
                continue
        else:
            if pending["type"] == "spring":
                reclaim_level = pending["bl"] + theta.m_bw * pending["bw"]
                if close >= reclaim_level:
                    entry = reclaim_level
                    stop = pending["bl"] - 0.4 * pending["atr"]
                    tp1 = (pending["bl"] + pending["bh"]) / 2.0
                    tp2 = pending["bh"]
                    signals.append(EventSignal(
                        t_signal=df2.index[i],
                        side="long",
                        event="spring",
                        entry_price=float(entry),
                        stop_price=float(stop),
                        tp1=float(tp1),
                        tp2=float(tp2),
                        meta={"break_time": df2.index[pending["break_i"]], "box_low": pending["bl"], "box_high": pending["bh"]}
                    ))
                    pending = None
            else:
                reclaim_level = pending["bh"] - theta.m_bw * pending["bw"]
                if close <= reclaim_level:
                    entry = reclaim_level
                    stop = pending["bh"] + 0.4 * pending["atr"]
                    tp1 = (pending["bl"] + pending["bh"]) / 2.0
                    tp2 = pending["bl"]
                    signals.append(EventSignal(
                        t_signal=df2.index[i],
                        side="short",
                        event="utad",
                        entry_price=float(entry),
                        stop_price=float(stop),
                        tp1=float(tp1),
                        tp2=float(tp2),
                        meta={"break_time": df2.index[pending["break_i"]], "box_low": pending["bl"], "box_high": pending["bh"]}
                    ))
                    pending = None

    return signals
