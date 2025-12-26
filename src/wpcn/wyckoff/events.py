from __future__ import annotations
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from wpcn.core.types import Theta, EventSignal
from wpcn.features.indicators import atr, detect_rsi_divergence, detect_fib_bounce, rsi, stoch_rsi
from wpcn.wyckoff.box import box_engine_freeze

def detect_spring_utad(df: pd.DataFrame, theta: Theta) -> List[EventSignal]:
    _atr = atr(df, theta.atr_len)
    box = box_engine_freeze(df, theta)
    df2 = df.join(_atr.rename("atr")).join(box)

    signals: List[EventSignal] = []
    pending: Optional[Dict[str, Any]] = None

    # OPTIMIZATION: Pre-extract numpy arrays for faster iteration
    idx = df2.index
    close_arr = df2["close"].values
    low_arr = df2["low"].values
    high_arr = df2["high"].values
    atr_arr = df2["atr"].values
    bl_arr = df2["box_low"].values
    bh_arr = df2["box_high"].values
    bw_arr = df2["box_width"].values
    n = len(df2)

    for i in range(n):
        # Check for NaN using numpy (faster than pandas)
        if np.isnan(atr_arr[i]) or np.isnan(bl_arr[i]) or np.isnan(bh_arr[i]) or np.isnan(bw_arr[i]):
            continue

        bl, bh, bw = bl_arr[i], bh_arr[i], bw_arr[i]
        at = atr_arr[i]
        close, low, high = close_arr[i], low_arr[i], high_arr[i]

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
                        t_signal=idx[i],
                        side="long",
                        event="spring",
                        entry_price=float(entry),
                        stop_price=float(stop),
                        tp1=float(tp1),
                        tp2=float(tp2),
                        meta={"break_time": idx[pending["break_i"]], "box_low": pending["bl"], "box_high": pending["bh"]}
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
                        t_signal=idx[i],
                        side="short",
                        event="utad",
                        entry_price=float(entry),
                        stop_price=float(stop),
                        tp1=float(tp1),
                        tp2=float(tp2),
                        meta={"break_time": idx[pending["break_i"]], "box_low": pending["bl"], "box_high": pending["bh"]}
                    ))
                    pending = None

    return signals


def detect_scalping_signals(df: pd.DataFrame, theta: Theta) -> List[EventSignal]:
    """
    Enhanced scalping signal detection using:
    1. RSI Divergence (bullish/bearish)
    2. Fibonacci retracement bounces
    3. Stochastic RSI oversold/overbought
    4. Original Wyckoff Spring/UTAD

    Designed for higher frequency trading (10+ trades/day on 15m).
    """
    signals: List[EventSignal] = []

    # Calculate indicators
    _atr = atr(df, theta.atr_len)
    div_df = detect_rsi_divergence(df, rsi_len=14, lookback=2)  # Even shorter lookback for more signals
    fib_df = detect_fib_bounce(df, swing_lookback=8, tolerance=0.008)  # Wider tolerance for more bounces
    k, d = stoch_rsi(df['close'], 14, 14, 3, 3)
    _rsi = rsi(df['close'], 14)

    # Pre-extract arrays for speed
    idx = df.index
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr_arr = _atr.values
    rsi_arr = _rsi.values
    stoch_k_arr = k.values
    stoch_d_arr = d.values

    bullish_div = div_df['bullish_div'].values
    bearish_div = div_df['bearish_div'].values
    fib_long = fib_df['fib_bounce_long'].values
    fib_short = fib_df['fib_bounce_short'].values
    fib_level = fib_df['fib_level'].values

    n = len(df)

    # Cooldown to prevent overtrading (minimum bars between signals)
    # 15m = 4 bars = 1시간 쿨다운 (적절한 빈도)
    last_signal_i = -10

    for i in range(15, n):
        if np.isnan(atr_arr[i]) or np.isnan(rsi_arr[i]):
            continue

        # Cooldown check (at least 4 bars = 1시간 between signals for scalping)
        if i - last_signal_i < 4:
            continue

        at = atr_arr[i]
        close = close_arr[i]
        high = high_arr[i]
        low = low_arr[i]
        current_rsi = rsi_arr[i]
        current_k = stoch_k_arr[i]

        # === LONG SIGNALS ===

        # 1. RSI Bullish Divergence + Oversold (더 완화: RSI < 50)
        if bullish_div[i] and current_rsi < 50:
            entry = close
            stop = low - 0.4 * at
            tp1 = close + 0.8 * at
            tp2 = close + 1.5 * at
            signals.append(EventSignal(
                t_signal=idx[i],
                side="long",
                event="rsi_div_long",
                entry_price=float(entry),
                stop_price=float(stop),
                tp1=float(tp1),
                tp2=float(tp2),
                meta={"rsi": current_rsi, "signal_type": "divergence"}
            ))
            last_signal_i = i
            continue

        # 2. Fibonacci Bounce Long + Stoch RSI Oversold (더 완화: K < 0.50)
        if fib_long[i] and current_k < 0.50:
            entry = close
            stop = low - 0.4 * at
            tp1 = close + 0.8 * at
            tp2 = close + 1.2 * at
            signals.append(EventSignal(
                t_signal=idx[i],
                side="long",
                event="fib_bounce_long",
                entry_price=float(entry),
                stop_price=float(stop),
                tp1=float(tp1),
                tp2=float(tp2),
                meta={"fib_level": fib_level[i], "stoch_k": current_k, "signal_type": "fibonacci"}
            ))
            last_signal_i = i
            continue

        # 3. Stoch RSI Oversold Reversal (더 완화: K < 0.40, 반등 시작)
        if i > 0 and current_k < 0.40 and current_k > stoch_k_arr[i-1] and stoch_k_arr[i-1] < 0.30:
            entry = close
            stop = low - 0.6 * at
            tp1 = close + 0.8 * at
            tp2 = close + 1.5 * at
            signals.append(EventSignal(
                t_signal=idx[i],
                side="long",
                event="stoch_cross_long",
                entry_price=float(entry),
                stop_price=float(stop),
                tp1=float(tp1),
                tp2=float(tp2),
                meta={"stoch_k": current_k, "stoch_d": stoch_d_arr[i], "signal_type": "stochastic"}
            ))
            last_signal_i = i
            continue

        # === SHORT SIGNALS ===

        # 1. RSI Bearish Divergence + Overbought (더 완화: RSI > 50)
        if bearish_div[i] and current_rsi > 50:
            entry = close
            stop = high + 0.4 * at
            tp1 = close - 0.8 * at
            tp2 = close - 1.5 * at
            signals.append(EventSignal(
                t_signal=idx[i],
                side="short",
                event="rsi_div_short",
                entry_price=float(entry),
                stop_price=float(stop),
                tp1=float(tp1),
                tp2=float(tp2),
                meta={"rsi": current_rsi, "signal_type": "divergence"}
            ))
            last_signal_i = i
            continue

        # 2. Fibonacci Bounce Short + Stoch RSI Overbought (더 완화: K > 0.50)
        if fib_short[i] and current_k > 0.50:
            entry = close
            stop = high + 0.4 * at
            tp1 = close - 0.8 * at
            tp2 = close - 1.2 * at
            signals.append(EventSignal(
                t_signal=idx[i],
                side="short",
                event="fib_bounce_short",
                entry_price=float(entry),
                stop_price=float(stop),
                tp1=float(tp1),
                tp2=float(tp2),
                meta={"fib_level": fib_level[i], "stoch_k": current_k, "signal_type": "fibonacci"}
            ))
            last_signal_i = i
            continue

        # 3. Stoch RSI Overbought Reversal (더 완화: K > 0.60, 하락 시작)
        if i > 0 and current_k > 0.60 and current_k < stoch_k_arr[i-1] and stoch_k_arr[i-1] > 0.70:
            entry = close
            stop = high + 0.5 * at
            tp1 = close - 0.8 * at
            tp2 = close - 1.5 * at
            signals.append(EventSignal(
                t_signal=idx[i],
                side="short",
                event="stoch_cross_short",
                entry_price=float(entry),
                stop_price=float(stop),
                tp1=float(tp1),
                tp2=float(tp2),
                meta={"stoch_k": current_k, "stoch_d": stoch_d_arr[i], "signal_type": "stochastic"}
            ))
            last_signal_i = i
            continue

    return signals


def detect_all_signals(df: pd.DataFrame, theta: Theta, scalping_mode: bool = False) -> List[EventSignal]:
    """
    Combined signal detection.

    If scalping_mode is True, uses the enhanced scalping signals.
    Otherwise, uses the original Wyckoff Spring/UTAD signals.
    """
    if scalping_mode:
        # Get both scalping signals and original signals
        scalping_signals = detect_scalping_signals(df, theta)
        wyckoff_signals = detect_spring_utad(df, theta)

        # Combine and sort by time
        all_signals = scalping_signals + wyckoff_signals
        all_signals.sort(key=lambda s: s.t_signal)

        # Remove duplicate signals at the same time (keep first one)
        seen_times = set()
        unique_signals = []
        for sig in all_signals:
            if sig.t_signal not in seen_times:
                seen_times.add(sig.t_signal)
                unique_signals.append(sig)

        return unique_signals
    else:
        return detect_spring_utad(df, theta)
