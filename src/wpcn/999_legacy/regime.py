from __future__ import annotations
import numpy as np
import pandas as pd
from wpcn.features.indicators import atr, adx, slope

def compute_regime(df: pd.DataFrame, atr_len: int, adx_len: int, adx_trend: float, chaos_ret_atr: float) -> pd.Series:
    at = atr(df, atr_len)
    ax = adx(df, adx_len)
    sl = slope(df["close"], 20)

    ret = df["close"].pct_change().fillna(0.0).abs()
    chaos = ret > (chaos_ret_atr * (at / (df["close"].abs() + 1e-12)).fillna(0.0))

    regime = pd.Series(index=df.index, dtype="object")
    regime.loc[:] = "RANGE"
    regime.loc[chaos] = "CHAOS"
    regime.loc[(~chaos) & (ax >= adx_trend) & (sl > 0)] = "TREND_UP"
    regime.loc[(~chaos) & (ax >= adx_trend) & (sl < 0)] = "TREND_DOWN"
    return regime
