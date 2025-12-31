from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd

from wpcn._03_common._01_core.types import Theta, BacktestConfig
from wpcn._02_data.resample import resample_ohlcv, merge_asof_back
from wpcn._02_data.timeframes import timeframe_to_pandas_rule
from wpcn._03_common._02_features.indicators import atr, zscore, rsi, stoch_rsi
from wpcn._03_common._02_features.targets import compute_target_candidates
from wpcn._03_common._09_gate.regime import compute_regime
from wpcn._03_common._03_wyckoff.box import box_engine_freeze
from wpcn._03_common._03_wyckoff.events import detect_spring_utad

PAGES = ["accum", "reaccum", "distrib", "redistr", "spring", "utad"]

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.nanmax(x)
    e = np.exp(x)
    return e / (np.nansum(e) + 1e-12)

def compute_navigation(df: pd.DataFrame, theta: Theta, bt: BacktestConfig, mtf: List[str]) -> pd.DataFrame:
    df = df.sort_index()

    # base features
    at = atr(df, theta.atr_len)
    box = box_engine_freeze(df, theta)
    z = zscore(df["close"], 20)
    r = rsi(df["close"], 14)
    k, d = stoch_rsi(df["close"], 14, 14, 3, 3)

    # effort vs result (abs body scaled by volume ratio)
    vol_ma = df["volume"].rolling(20, min_periods=20).mean()
    eff = (df["close"] - df["open"]).abs() / ((df["volume"] / (vol_ma + 1e-12)) + 1e-12)
    eff_q = eff.rolling(200, min_periods=50).quantile(0.25)  # low-eff threshold proxy

    regime = compute_regime(df, theta.atr_len, bt.adx_len, bt.adx_trend, bt.chaos_ret_atr)

    # MTF fusion: resample, compute HTF zscore + RSI, merge back as-of
    mtf_feats = []
    for tf in mtf or []:
        rule = timeframe_to_pandas_rule(tf)
        htf = resample_ohlcv(df, rule)
        htf_z = zscore(htf["close"], 20).rename(f"z_{tf}")
        htf_r = rsi(htf["close"], 14).rename(f"rsi_{tf}")
        merged = merge_asof_back(df.index, pd.concat([htf_z, htf_r], axis=1))
        mtf_feats.append(merged)

    mtf_df = pd.concat(mtf_feats, axis=1) if mtf_feats else pd.DataFrame(index=df.index)

    # edge events (binary at signal time)
    signals = detect_spring_utad(df, theta)
    edge = pd.Series(0.0, index=df.index)
    edge_event = pd.Series("", index=df.index, dtype="object")
    for s in signals:
        edge.loc[s.t_signal] = 1.0
        edge_event.loc[s.t_signal] = s.event

    out = pd.DataFrame(index=df.index)
    out["regime"] = regime
    out["edge_score"] = edge
    out["edge_event"] = edge_event
    out = out.join(pd.concat([at.rename("atr"), box, z.rename("z"), r.rename("rsi"), k.rename("stoch_k"), d.rename("stoch_d"), eff.rename("eff"), eff_q.rename("eff_q")], axis=1))
    out = out.join(mtf_df)

    # target candidates
    target_df = compute_target_candidates(df, theta, bt)
    out = out.join(target_df)

    # scoring (heuristic, designed to be learned/calibrated later)
    scores = np.zeros((len(out), len(PAGES)), dtype=float)
    for i, t in enumerate(out.index):
        row = out.iloc[i]
        if row["regime"] == "CHAOS" or np.isnan(row.get("box_width", np.nan)):
            scores[i, :] = -10.0
            continue

        # absorption: low-effort indicates absorption/exhaustion
        absorption = 1.0 if (not np.isnan(row["eff_q"]) and row["eff"] <= row["eff_q"]) else 0.0
        z_ext_low = 1.0 if (not np.isnan(row["z"]) and row["z"] <= -2.0) else 0.0
        z_ext_high = 1.0 if (not np.isnan(row["z"]) and row["z"] >= 2.0) else 0.0

        # HTF confirmation: any htf oversold/overbought
        htf_oversold = 0.0
        htf_overbought = 0.0
        for tf in mtf or []:
            zv = row.get(f"z_{tf}", np.nan)
            if not np.isnan(zv):
                if zv <= -1.5: htf_oversold = 1.0
                if zv >= 1.5: htf_overbought = 1.0

        is_range = 1.0 if row["regime"] == "RANGE" else 0.0
        is_up = 1.0 if row["regime"] == "TREND_UP" else 0.0
        is_dn = 1.0 if row["regime"] == "TREND_DOWN" else 0.0

        # Base pages
        scores[i, PAGES.index("accum")] = 1.2*is_range + 1.0*z_ext_low + 0.7*absorption + 0.6*htf_oversold
        scores[i, PAGES.index("distrib")] = 1.2*is_range + 1.0*z_ext_high + 0.7*absorption + 0.6*htf_overbought
        scores[i, PAGES.index("reaccum")] = 1.0*is_up + 0.4*is_range + 0.3*htf_oversold
        scores[i, PAGES.index("redistr")] = 1.0*is_dn + 0.4*is_range + 0.3*htf_overbought

        # Event pages get a big boost on edge bars
        ev = row["edge_event"]
        if ev == "spring":
            scores[i, PAGES.index("spring")] = 4.0 + 1.0*absorption + 0.6*htf_oversold
        else:
            scores[i, PAGES.index("spring")] = -0.5 + 0.3*z_ext_low
        if ev == "utad":
            scores[i, PAGES.index("utad")] = 4.0 + 1.0*absorption + 0.6*htf_overbought
        else:
            scores[i, PAGES.index("utad")] = -0.5 + 0.3*z_ext_high

    # softmax per row
    probs = np.zeros_like(scores)
    for i in range(len(scores)):
        probs[i] = _softmax(scores[i])

    for j, name in enumerate(PAGES):
        out[f"p_{name}"] = probs[:, j]

    out["top1_page"] = pd.Series([PAGES[int(np.argmax(probs[i]))] for i in range(len(probs))], index=out.index)
    out["confidence"] = pd.Series([float(np.max(probs[i])) for i in range(len(probs))], index=out.index)

    # confirm bars
    top1 = out["top1_page"]
    stable = (top1 == top1.shift(1)).astype(int)
    # run length of same top1 (approx)
    runlen = []
    c = 0
    prev = None
    for v in top1.values:
        if v == prev:
            c += 1
        else:
            c = 1
            prev = v
        runlen.append(c)
    out["top1_runlen"] = runlen

    # edge_score 조건 포함한 원래 조건으로 복원
    out["trade_gate"] = (
        (out["regime"] != "CHAOS") &
        (out["confidence"] >= bt.conf_min) &
        (out["edge_score"] >= bt.edge_min) &
        (out["top1_runlen"] >= bt.confirm_bars)
    ).astype(int)

    return out
