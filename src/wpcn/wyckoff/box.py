from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from wpcn.core.types import Theta

@dataclass
class BoxState:
    high: float = np.nan
    low: float = np.nan
    mid: float = np.nan
    width: float = np.nan
    frozen_until_i: int = -1

def box_engine_freeze(df: pd.DataFrame, theta: Theta) -> pd.DataFrame:
    hi_roll = df["high"].rolling(theta.box_L, min_periods=theta.box_L).max()
    lo_roll = df["low"].rolling(theta.box_L, min_periods=theta.box_L).min()
    width_roll = (hi_roll - lo_roll)

    out = pd.DataFrame(index=df.index, columns=["box_high","box_low","box_mid","box_width"], dtype=float)

    state = BoxState()
    for i in range(len(df)):
        if np.isnan(hi_roll.iloc[i]) or np.isnan(lo_roll.iloc[i]):
            continue

        if i <= state.frozen_until_i:
            out.iloc[i] = [state.high, state.low, state.mid, state.width]
            continue

        bh = float(hi_roll.iloc[i])
        bl = float(lo_roll.iloc[i])
        bw = float(width_roll.iloc[i])
        bm = (bh + bl) / 2.0

        med_bw = float(width_roll.iloc[max(0, i-theta.box_L):i+1].median())
        box_candidate = (bw <= 1.5 * med_bw) if med_bw > 0 else False

        if box_candidate:
            state = BoxState(high=bh, low=bl, mid=bm, width=bw, frozen_until_i=i + theta.m_freeze)
        else:
            state = BoxState(high=bh, low=bl, mid=bm, width=bw, frozen_until_i=-1)

        out.iloc[i] = [state.high, state.low, state.mid, state.width]

    return out
