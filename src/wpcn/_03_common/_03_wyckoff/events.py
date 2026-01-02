from __future__ import annotations
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from wpcn.core.types import Theta, EventSignal
from wpcn.features.indicators import atr, detect_rsi_divergence, detect_fib_bounce, rsi, stoch_rsi
from wpcn.features.dynamic_params import DynamicParameterEngine, DynamicParams
from wpcn.wyckoff.box import box_engine_freeze

def detect_spring_utad(
    df: pd.DataFrame,
    theta: Theta,
    use_trend_filter: bool = True,
    reclaim_hold_bars: int = 2
) -> List[EventSignal]:
    """
    Wyckoff Spring/UTAD 신호 감지 (추세 필터 + reclaim 유지 조건 적용)

    - Spring: 박스권 하단 돌파 후 회복 → 롱 (하락 추세에서는 무시)
    - UTAD: 박스권 상단 돌파 후 회복 → 숏 (상승 추세에서는 무시)

    reclaim_hold_bars: reclaim 후 박스 내부 유지해야 하는 봉 수 (휩쏘 방지)
    - reclaim만 하고 다음 봉에 다시 박스 밖으로 빠지면 무효화
    """
    _atr = atr(df, theta.atr_len)
    box = box_engine_freeze(df, theta)
    df2 = df.join(_atr.rename("atr")).join(box)

    signals: List[EventSignal] = []
    pending: Optional[Dict[str, Any]] = None

    # 추세 감지를 위한 동적 파라미터 엔진
    if use_trend_filter:
        dyn_engine = DynamicParameterEngine(vol_lookback=100)
        dyn_params_df = dyn_engine.compute_params_series(df, min_lookback=50)
    else:
        dyn_params_df = None

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

        # 추세 필터 적용
        if use_trend_filter and dyn_params_df is not None and idx[i] in dyn_params_df.index:
            dp = dyn_params_df.loc[idx[i]]
            trend_dir = dp['trend_direction']
            trend_str = dp['trend_strength']
            stop_mult = dp['atr_mult_stop']
        else:
            trend_dir = 'neutral'
            trend_str = 0.0
            stop_mult = 0.4

        # 추세 기반 필터링
        if trend_str > 0.5:
            allow_long = trend_dir != 'bearish'  # 강한 하락 추세면 롱 금지
            allow_short = trend_dir != 'bullish'  # 강한 상승 추세면 숏 금지
        elif trend_str > 0.3:
            allow_long = True
            allow_short = trend_dir != 'bullish'
        else:
            allow_long = True
            allow_short = True

        # expire pending break
        if pending is not None and i > pending["break_i"] + theta.N_reclaim + reclaim_hold_bars:
            pending = None

        # new break
        if pending is None:
            if low < bl - theta.x_atr * at:
                pending = {"type":"spring", "break_i": i, "bl": bl, "bh": bh, "bw": bw, "atr": at, "reclaim_i": None}
                continue
            if high > bh + theta.x_atr * at:
                pending = {"type":"utad", "break_i": i, "bl": bl, "bh": bh, "bw": bw, "atr": at, "reclaim_i": None}
                continue
        else:
            if pending["type"] == "spring" and allow_long:
                reclaim_level = pending["bl"] + theta.m_bw * pending["bw"]

                # Step 1: reclaim 감지
                if pending["reclaim_i"] is None and close >= reclaim_level:
                    pending["reclaim_i"] = i
                    continue

                # Step 2: reclaim 후 유지 확인 (reclaim_hold_bars 동안 박스 내부 유지)
                if pending["reclaim_i"] is not None:
                    # 박스 밖으로 다시 빠지면 무효화 (휩쏘)
                    if low < pending["bl"] - theta.x_atr * pending["atr"]:
                        pending = None  # 휩쏘 - 신호 무효화
                        continue

                    # reclaim_hold_bars 동안 유지 확인
                    bars_since_reclaim = i - pending["reclaim_i"]
                    if bars_since_reclaim >= reclaim_hold_bars:
                        entry = reclaim_level
                        stop = pending["bl"] - stop_mult * pending["atr"]
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
                            meta={
                                "break_time": idx[pending["break_i"]],
                                "reclaim_time": idx[pending["reclaim_i"]],
                                "hold_bars": bars_since_reclaim,
                                "box_low": pending["bl"],
                                "box_high": pending["bh"],
                                "trend": trend_dir,
                                "trend_strength": trend_str
                            }
                        ))
                        pending = None

            elif pending["type"] == "utad" and allow_short:
                reclaim_level = pending["bh"] - theta.m_bw * pending["bw"]

                # Step 1: reclaim 감지
                if pending["reclaim_i"] is None and close <= reclaim_level:
                    pending["reclaim_i"] = i
                    continue

                # Step 2: reclaim 후 유지 확인
                if pending["reclaim_i"] is not None:
                    # 박스 밖으로 다시 빠지면 무효화 (휩쏘)
                    if high > pending["bh"] + theta.x_atr * pending["atr"]:
                        pending = None  # 휩쏘 - 신호 무효화
                        continue

                    # reclaim_hold_bars 동안 유지 확인
                    bars_since_reclaim = i - pending["reclaim_i"]
                    if bars_since_reclaim >= reclaim_hold_bars:
                        entry = reclaim_level
                        stop = pending["bh"] + stop_mult * pending["atr"]
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
                            meta={
                                "break_time": idx[pending["break_i"]],
                                "reclaim_time": idx[pending["reclaim_i"]],
                                "hold_bars": bars_since_reclaim,
                                "box_low": pending["bl"],
                                "box_high": pending["bh"],
                                "trend": trend_dir,
                                "trend_strength": trend_str
                            }
                        ))
                        pending = None

            elif pending["type"] == "spring" and not allow_long:
                # 하락 추세에서 Spring 무시
                pending = None
            elif pending["type"] == "utad" and not allow_short:
                # 상승 추세에서 UTAD 무시
                pending = None

    return signals


def detect_scalping_signals_dynamic(df: pd.DataFrame, theta: Theta, use_dynamic: bool = True) -> List[EventSignal]:
    """
    Enhanced scalping signal detection with DYNAMIC PARAMETERS.

    Uses Maxwell-Boltzmann distribution for volatility modeling,
    FFT for cycle detection, and Hilbert transform for phase analysis.

    This allows the strategy to adapt to different coins and market conditions.

    FIXED (P0-1): Stoch RSI 룩어헤드 버그 수정
    - shift(1)을 적용하여 완성된 봉의 Stoch RSI만 사용
    - 현재 봉은 아직 완성되지 않았으므로 이전 봉 값을 참조해야 함
    """
    signals: List[EventSignal] = []

    # Calculate indicators
    _atr = atr(df, theta.atr_len)
    div_df = detect_rsi_divergence(df, rsi_len=14, lookback=2)
    fib_df = detect_fib_bounce(df, swing_lookback=8, tolerance=0.008)
    k, d = stoch_rsi(df['close'], 14, 14, 3, 3)
    _rsi = rsi(df['close'], 14)

    # FIXED (P0-1): Stoch RSI에 shift(1) 적용하여 완성된 봉만 참조
    # 현재 봉의 stoch는 아직 봉이 완성되지 않아 변할 수 있으므로,
    # 이전 봉의 확정된 stoch 값을 사용
    k = k.shift(1)
    d = d.shift(1)

    # Initialize dynamic parameter engine
    if use_dynamic:
        dyn_engine = DynamicParameterEngine(
            vol_lookback=100,
            fft_min_period=5,
            fft_max_period=100,
            hilbert_smooth=5
        )
        # Pre-compute dynamic params for all bars (for efficiency)
        dyn_params_df = dyn_engine.compute_params_series(df, min_lookback=50)
        print(f"[Dynamic] Volatility regimes detected, FFT cycles analyzed")
    else:
        dyn_params_df = None

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

    # Cooldown: dynamically adjusted based on dominant cycle
    base_cooldown = 4
    last_signal_i = -10

    for i in range(50, n):  # Start from 50 for dynamic params
        if np.isnan(atr_arr[i]) or np.isnan(rsi_arr[i]):
            continue

        # Get dynamic parameters for this bar
        if use_dynamic and dyn_params_df is not None and idx[i] in dyn_params_df.index:
            dp = dyn_params_df.loc[idx[i]]
            stop_mult = dp['atr_mult_stop']
            tp1_mult = dp['atr_mult_tp1']
            tp2_mult = dp['atr_mult_tp2']
            vol_regime = dp['volatility_regime']
            phase_pos = dp['phase_position']
            trend_dir = dp['trend_direction']
            trend_str = dp['trend_strength']
            confidence = dp['confidence']
            dominant_cycle = dp['dominant_cycle']

            # Dynamic cooldown: shorter in high volatility, longer in low
            if vol_regime == 'extreme':
                cooldown = max(2, base_cooldown // 2)
            elif vol_regime == 'high':
                cooldown = base_cooldown
            elif vol_regime == 'medium':
                cooldown = base_cooldown + 2
            else:  # low
                cooldown = base_cooldown + 4
        else:
            # Fallback to static parameters
            stop_mult = 0.4
            tp1_mult = 0.8
            tp2_mult = 1.5
            vol_regime = 'medium'
            phase_pos = 'unknown'
            trend_dir = 'neutral'
            trend_str = 0.0
            confidence = 0.5
            dominant_cycle = 20
            cooldown = base_cooldown

        # Cooldown check
        if i - last_signal_i < cooldown:
            continue

        at = atr_arr[i]
        close = close_arr[i]
        high = high_arr[i]
        low = low_arr[i]
        current_rsi = rsi_arr[i]
        current_k = stoch_k_arr[i]

        # === TREND-BASED SIGNAL FILTERING (핵심!) ===
        # 상승 추세에서는 롱만, 하락 추세에서는 숏만
        # 강한 추세(strength > 0.5)일 때 더 엄격하게 필터링
        if trend_str > 0.5:
            # 강한 상승 추세: 숏 금지
            if trend_dir == 'bullish':
                long_trend_ok = True
                short_trend_ok = False
            # 강한 하락 추세: 롱 금지
            elif trend_dir == 'bearish':
                long_trend_ok = False
                short_trend_ok = True
            else:
                long_trend_ok = True
                short_trend_ok = True
        elif trend_str > 0.3:
            # 중간 추세: 역추세 신호 신뢰도 낮춤
            long_trend_ok = True
            short_trend_ok = trend_dir != 'bullish'  # 상승 추세면 숏 비추
        else:
            # 약한 추세/횡보: 양방향 허용
            long_trend_ok = True
            short_trend_ok = True

        # === PHASE-BASED SIGNAL FILTERING ===
        long_phase_ok = phase_pos in ['bottom', 'rising', 'unknown']
        short_phase_ok = phase_pos in ['top', 'falling', 'unknown']

        # 최종 필터: 추세 + 위상 모두 OK해야 함
        long_ok = long_trend_ok and long_phase_ok
        short_ok = short_trend_ok and short_phase_ok

        # === LONG SIGNALS ===

        # 1. RSI Bullish Divergence + Oversold
        if bullish_div[i] and current_rsi < 50 and long_ok:
            entry = close
            stop = low - stop_mult * at
            tp1 = close + tp1_mult * at
            tp2 = close + tp2_mult * at
            signals.append(EventSignal(
                t_signal=idx[i],
                side="long",
                event="rsi_div_long",
                entry_price=float(entry),
                stop_price=float(stop),
                tp1=float(tp1),
                tp2=float(tp2),
                meta={
                    "rsi": current_rsi,
                    "signal_type": "divergence",
                    "vol_regime": vol_regime,
                    "phase": phase_pos,
                    "trend": trend_dir,
                    "trend_strength": trend_str,
                    "confidence": confidence,
                    "dynamic": use_dynamic
                }
            ))
            last_signal_i = i
            continue

        # 2. Fibonacci Bounce Long + Stoch RSI Oversold
        # FIXED (P0-1): NaN 체크 추가 (shift로 인해 처음 몇 봉은 NaN)
        if fib_long[i] and not np.isnan(current_k) and current_k < 0.50 and long_ok:
            entry = close
            stop = low - stop_mult * at
            tp1 = close + tp1_mult * at
            tp2 = close + (tp1_mult + 0.4) * at
            signals.append(EventSignal(
                t_signal=idx[i],
                side="long",
                event="fib_bounce_long",
                entry_price=float(entry),
                stop_price=float(stop),
                tp1=float(tp1),
                tp2=float(tp2),
                meta={
                    "fib_level": fib_level[i],
                    "stoch_k": current_k,
                    "signal_type": "fibonacci",
                    "vol_regime": vol_regime,
                    "phase": phase_pos,
                    "trend": trend_dir,
                    "trend_strength": trend_str,
                    "confidence": confidence,
                    "dynamic": use_dynamic
                }
            ))
            last_signal_i = i
            continue

        # 3. Stoch RSI Oversold Reversal
        # FIXED (P0-1): shift(1)이 적용되어 stoch_k_arr[i]가 i-1봉 값이므로,
        # stoch_k_arr[i-1]은 i-2봉 값. 크로스 확인용으로 그대로 사용 가능
        # NaN 체크 추가
        prev_k = stoch_k_arr[i-1] if i > 0 and not np.isnan(stoch_k_arr[i-1]) else np.nan
        if i > 1 and not np.isnan(current_k) and not np.isnan(prev_k) and current_k < 0.40 and current_k > prev_k and prev_k < 0.30 and long_ok:
            entry = close
            stop = low - (stop_mult + 0.2) * at
            tp1 = close + tp1_mult * at
            tp2 = close + tp2_mult * at
            signals.append(EventSignal(
                t_signal=idx[i],
                side="long",
                event="stoch_cross_long",
                entry_price=float(entry),
                stop_price=float(stop),
                tp1=float(tp1),
                tp2=float(tp2),
                meta={
                    "stoch_k": current_k,
                    "stoch_d": stoch_d_arr[i],
                    "signal_type": "stochastic",
                    "vol_regime": vol_regime,
                    "phase": phase_pos,
                    "trend": trend_dir,
                    "trend_strength": trend_str,
                    "confidence": confidence,
                    "dynamic": use_dynamic
                }
            ))
            last_signal_i = i
            continue

        # === SHORT SIGNALS ===

        # 1. RSI Bearish Divergence + Overbought
        if bearish_div[i] and current_rsi > 50 and short_ok:
            entry = close
            stop = high + stop_mult * at
            tp1 = close - tp1_mult * at
            tp2 = close - tp2_mult * at
            signals.append(EventSignal(
                t_signal=idx[i],
                side="short",
                event="rsi_div_short",
                entry_price=float(entry),
                stop_price=float(stop),
                tp1=float(tp1),
                tp2=float(tp2),
                meta={
                    "rsi": current_rsi,
                    "signal_type": "divergence",
                    "vol_regime": vol_regime,
                    "phase": phase_pos,
                    "trend": trend_dir,
                    "trend_strength": trend_str,
                    "confidence": confidence,
                    "dynamic": use_dynamic
                }
            ))
            last_signal_i = i
            continue

        # 2. Fibonacci Bounce Short + Stoch RSI Overbought
        # FIXED (P0-1): NaN 체크 추가
        if fib_short[i] and not np.isnan(current_k) and current_k > 0.50 and short_ok:
            entry = close
            stop = high + stop_mult * at
            tp1 = close - tp1_mult * at
            tp2 = close - (tp1_mult + 0.4) * at
            signals.append(EventSignal(
                t_signal=idx[i],
                side="short",
                event="fib_bounce_short",
                entry_price=float(entry),
                stop_price=float(stop),
                tp1=float(tp1),
                tp2=float(tp2),
                meta={
                    "fib_level": fib_level[i],
                    "stoch_k": current_k,
                    "signal_type": "fibonacci",
                    "vol_regime": vol_regime,
                    "phase": phase_pos,
                    "trend": trend_dir,
                    "trend_strength": trend_str,
                    "confidence": confidence,
                    "dynamic": use_dynamic
                }
            ))
            last_signal_i = i
            continue

        # 3. Stoch RSI Overbought Reversal
        # FIXED (P0-1): shift(1) 적용으로 NaN 체크 필요
        if i > 1 and not np.isnan(current_k) and not np.isnan(prev_k) and current_k > 0.60 and current_k < prev_k and prev_k > 0.70 and short_ok:
            entry = close
            stop = high + (stop_mult + 0.1) * at
            tp1 = close - tp1_mult * at
            tp2 = close - tp2_mult * at
            signals.append(EventSignal(
                t_signal=idx[i],
                side="short",
                event="stoch_cross_short",
                entry_price=float(entry),
                stop_price=float(stop),
                tp1=float(tp1),
                tp2=float(tp2),
                meta={
                    "stoch_k": current_k,
                    "stoch_d": stoch_d_arr[i],
                    "signal_type": "stochastic",
                    "vol_regime": vol_regime,
                    "phase": phase_pos,
                    "trend": trend_dir,
                    "trend_strength": trend_str,
                    "confidence": confidence,
                    "dynamic": use_dynamic
                }
            ))
            last_signal_i = i
            continue

    return signals


def detect_scalping_signals(df: pd.DataFrame, theta: Theta) -> List[EventSignal]:
    """
    Wrapper for backward compatibility.
    Uses dynamic parameters by default.
    """
    return detect_scalping_signals_dynamic(df, theta, use_dynamic=True)


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
