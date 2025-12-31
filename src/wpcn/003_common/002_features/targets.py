from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from wpcn.features.indicators import rsi, stoch_rsi, atr
from wpcn.wyckoff.box import box_engine_freeze
from wpcn.wyckoff.events import detect_spring_utad
from wpcn.gate.regime import compute_regime

# Fibonacci levels
RETRACEMENT_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
EXTENSION_LEVELS = [1.0, 1.272, 1.618]

@dataclass
class TargetCandidate:
    level: float
    type: str  # 'support' or 'resistance'
    fib_ratio: float
    anchor_high: float
    anchor_low: float
    probability: float
    context: str  # e.g., 'accum_phase_c', 'trend_up'

def find_wyckoff_anchors(df: pd.DataFrame, theta, max_anchors: int = 50) -> List[Dict[str, Any]]:
    """Find Wyckoff event-based anchors (SC/AR for accum, BC/AR for distrib).

    OPTIMIZATION: Limit to recent anchors only to reduce O(n^2) complexity.
    """
    box = box_engine_freeze(df, theta)
    anchors = []

    # OPTIMIZATION: Use numpy arrays for faster iteration
    box_high_arr = box['box_high'].values
    box_low_arr = box['box_low'].values
    idx = df.index
    n = len(box)

    # Only look at last portion of data for anchors
    start_idx = max(0, n - 200)  # Last 200 bars only

    for i in range(start_idx, n):
        if not np.isnan(box_high_arr[i]):
            anchors.append({
                'type': 'accum',
                'high': box_high_arr[i],
                'low': box_low_arr[i],
                'time': idx[i]
            })
            anchors.append({
                'type': 'distrib',
                'high': box_high_arr[i],
                'low': box_low_arr[i],
                'time': idx[i]
            })

            # Limit total anchors
            if len(anchors) >= max_anchors:
                break

    return anchors

def find_pivot_anchors(df: pd.DataFrame, lookback: int = 5, max_anchors: int = 30) -> List[Dict[str, Any]]:
    """Find pivot-based anchors.

    OPTIMIZATION: Limit to recent data and use vectorized operations.
    """
    anchors = []
    n = len(df)

    # Only look at recent portion
    start_idx = max(lookback, n - 200)
    end_idx = n - lookback

    if start_idx >= end_idx:
        return anchors

    # OPTIMIZATION: Use numpy arrays
    highs = df['high'].values
    lows = df['low'].values
    idx = df.index

    for i in range(start_idx, end_idx):
        window_high = highs[i-lookback:i+lookback+1]
        window_low = lows[i-lookback:i+lookback+1]

        if highs[i] == window_high.max():
            anchors.append({
                'type': 'pivot_high',
                'high': highs[i],
                'low': window_low.min(),
                'time': idx[i]
            })
            if len(anchors) >= max_anchors:
                break

        if lows[i] == window_low.min():
            anchors.append({
                'type': 'pivot_low',
                'high': window_high.max(),
                'low': lows[i],
                'time': idx[i]
            })
            if len(anchors) >= max_anchors:
                break

    return anchors

def generate_fib_targets(anchor: Dict[str, Any], context: str) -> List[TargetCandidate]:
    """Generate Fibonacci targets from an anchor."""
    targets = []
    high, low = anchor['high'], anchor['low']
    diff = high - low
    
    if context in ['accum', 'reaccum', 'trend_up']:
        # Support targets (retracements)
        for ratio in RETRACEMENT_LEVELS:
            level = high - diff * ratio
            targets.append(TargetCandidate(
                level=level, type='support', fib_ratio=ratio,
                anchor_high=high, anchor_low=low, probability=0.0, context=context
            ))
        # Resistance targets (extensions)
        for ratio in EXTENSION_LEVELS:
            level = high + diff * ratio
            targets.append(TargetCandidate(
                level=level, type='resistance', fib_ratio=ratio,
                anchor_high=high, anchor_low=low, probability=0.0, context=context
            ))
    elif context in ['distrib', 'redistrib', 'trend_down']:
        # Resistance targets (retracements)
        for ratio in RETRACEMENT_LEVELS:
            level = low + diff * ratio
            targets.append(TargetCandidate(
                level=level, type='resistance', fib_ratio=ratio,
                anchor_high=high, anchor_low=low, probability=0.0, context=context
            ))
        # Support targets (extensions)
        for ratio in EXTENSION_LEVELS:
            level = low - diff * ratio
            targets.append(TargetCandidate(
                level=level, type='support', fib_ratio=ratio,
                anchor_high=high, anchor_low=low, probability=0.0, context=context
            ))
    return targets

def score_target_probability(target: TargetCandidate, df: pd.DataFrame, row: pd.Series, theta) -> float:
    """Score target probability based on multiple factors."""
    price = row['close']
    dist = abs(price - target.level) / row.get('atr', 1.0)
    fib_proximity = max(0, 1 - dist / 0.5)  # Closer is better
    
    # RSI state
    rsi_val = row.get('rsi', 50)
    rsi_state = 0.0
    if target.type == 'support':
        if rsi_val < 30:
            rsi_state = 1.0  # Oversold
        elif rsi_val > 70:
            rsi_state = -0.5  # Overbought bad for support
    else:  # resistance
        if rsi_val > 70:
            rsi_state = 1.0  # Overbought
        elif rsi_val < 30:
            rsi_state = -0.5  # Oversold bad for resistance
    
    # Stoch RSI state
    stoch_k = row.get('stoch_k', 0.5)
    stoch_state = 0.0
    if target.type == 'support':
        if stoch_k < 0.2:
            stoch_state = 1.0  # Oversold
    else:
        if stoch_k > 0.8:
            stoch_state = 1.0  # Overbought
    
    # Divergence (simplified)
    divergence = 0.0  # Placeholder
    
    # Wyckoff context
    regime = row.get('regime', 'RANGE')
    wyckoff_context = 0.5
    if 'trend' in regime.lower():
        wyckoff_context = 1.0
    elif regime == 'CHAOS':
        wyckoff_context = 0.1
    
    # Volume absorption (simplified)
    volume_abs = 0.5  # Placeholder
    
    # Weighted score
    score = (
        0.2 * fib_proximity +
        0.2 * rsi_state +
        0.2 * stoch_state +
        0.1 * divergence +
        0.2 * wyckoff_context +
        0.1 * volume_abs
    )
    return max(0, min(1, score))  # Clamp to [0,1]

def compute_target_candidates(df: pd.DataFrame, theta, bt_config) -> pd.DataFrame:
    """Compute target candidates with probabilities.

    OPTIMIZATION: Reduced anchor generation and simplified scoring.
    """
    # Get anchors (limited count)
    wyckoff_anchors = find_wyckoff_anchors(df, theta, max_anchors=30)
    pivot_anchors = find_pivot_anchors(df, max_anchors=20)
    all_anchors = wyckoff_anchors + pivot_anchors

    # Get context (regime)
    regime = compute_regime(df, theta.atr_len, bt_config.adx_len, bt_config.adx_trend, bt_config.chaos_ret_atr)

    # Indicators - compute once
    at = atr(df, theta.atr_len)
    r = rsi(df['close'], 14)
    k, d = stoch_rsi(df['close'], 14, 14, 3, 3)

    # Generate targets
    all_targets = []
    for anchor in all_anchors:
        context = 'trend_up'
        targets = generate_fib_targets(anchor, context)
        all_targets.extend(targets)

    print(f"Generated {len(all_targets)} targets from {len(all_anchors)} anchors")

    # OPTIMIZATION: Pre-extract numpy arrays
    close_arr = df['close'].values
    atr_arr = at.values
    rsi_arr = r.values
    stoch_k_arr = k.values
    regime_arr = regime.values
    idx = df.index
    n = len(df)

    # Pre-extract target levels for vectorized distance calculation
    target_levels = np.array([t.level for t in all_targets]) if all_targets else np.array([])
    target_types = [t.type for t in all_targets]

    # Build result dictionary
    target_cols = {}

    for i in range(n):
        c = close_arr[i]
        if len(target_levels) == 0 or np.isnan(c):
            target_cols[idx[i]] = {
                'support_levels': [],
                'support_probs': [],
                'resist_levels': [],
                'resist_probs': []
            }
            continue

        # Vectorized distance calculation
        distances = np.abs(target_levels - c) / c

        # Filter relevant targets (within 10% for performance)
        mask = distances < 0.1
        if not mask.any():
            target_cols[idx[i]] = {
                'support_levels': [],
                'support_probs': [],
                'resist_levels': [],
                'resist_probs': []
            }
            continue

        # Get relevant indices
        relevant_indices = np.where(mask)[0]

        # Simplified scoring (avoid per-target function call)
        probs = np.maximum(0, 1 - distances[relevant_indices] / 0.1)

        # Sort by probability
        sorted_idx = np.argsort(-probs)[:6]  # Top 6

        top_support = []
        top_resist = []
        for j in sorted_idx:
            real_idx = relevant_indices[j]
            level = target_levels[real_idx]
            prob = probs[j]
            if target_types[real_idx] == 'support' and len(top_support) < 3:
                top_support.append((level, prob))
            elif target_types[real_idx] == 'resistance' and len(top_resist) < 3:
                top_resist.append((level, prob))

        target_cols[idx[i]] = {
            'support_levels': [s[0] for s in top_support],
            'support_probs': [s[1] for s in top_support],
            'resist_levels': [r[0] for r in top_resist],
            'resist_probs': [r[1] for r in top_resist]
        }

    # Convert to DataFrame
    target_df = pd.DataFrame.from_dict(target_cols, orient='index')
    return target_df