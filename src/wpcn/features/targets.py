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

def find_wyckoff_anchors(df: pd.DataFrame, theta) -> List[Dict[str, Any]]:
    """Find Wyckoff event-based anchors (SC/AR for accum, BC/AR for distrib)."""
    box = box_engine_freeze(df, theta)
    anchors = []
    
    # Simple proxy: box highs/lows as anchors
    for i in range(len(box)):
        if not np.isnan(box.iloc[i]['box_high']):
            # Accumulation anchor: SC (box_low) to AR (first rebound high)
            anchors.append({
                'type': 'accum',
                'high': box.iloc[i]['box_high'],
                'low': box.iloc[i]['box_low'],
                'time': df.index[i]
            })
            # Distribution anchor: BC (box_high) to AR (first decline low)
            anchors.append({
                'type': 'distrib',
                'high': box.iloc[i]['box_high'],
                'low': box.iloc[i]['box_low'],
                'time': df.index[i]
            })
    return anchors

def find_pivot_anchors(df: pd.DataFrame, lookback: int = 5) -> List[Dict[str, Any]]:
    """Find pivot-based anchors."""
    anchors = []
    highs = df['high']
    lows = df['low']
    
    for i in range(lookback, len(df) - lookback):
        if highs.iloc[i] == highs.iloc[i-lookback:i+lookback+1].max():
            # Pivot high
            anchors.append({
                'type': 'pivot_high',
                'high': highs.iloc[i],
                'low': lows.iloc[i-lookback:i+lookback+1].min(),
                'time': df.index[i]
            })
        if lows.iloc[i] == lows.iloc[i-lookback:i+lookback+1].min():
            # Pivot low
            anchors.append({
                'type': 'pivot_low',
                'high': highs.iloc[i-lookback:i+lookback+1].max(),
                'low': lows.iloc[i],
                'time': df.index[i]
            })
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
    """Compute target candidates with probabilities."""
    # Get anchors
    wyckoff_anchors = find_wyckoff_anchors(df, theta)
    pivot_anchors = find_pivot_anchors(df)
    all_anchors = wyckoff_anchors + pivot_anchors
    
    # Get context (regime)
    regime = compute_regime(df, theta.atr_len, bt_config.adx_len, bt_config.adx_trend, bt_config.chaos_ret_atr)
    
    # Indicators
    at = atr(df, theta.atr_len)
    r = rsi(df['close'], 14)
    k, d = stoch_rsi(df['close'], 14, 14, 3, 3)
    
    # Generate targets
    all_targets = []
    for anchor in all_anchors:
        context = 'trend_up'  # Hardcode for testing
        targets = generate_fib_targets(anchor, context)
        all_targets.extend(targets)
    
    print(f"Generated {len(all_targets)} targets from {len(all_anchors)} anchors")  # Debug
    
    # Score at each bar
    target_cols = {}
    for i, t in enumerate(df.index):
        row = df.iloc[i].copy()
        row['atr'] = at.iloc[i]
        row['rsi'] = r.iloc[i]
        row['stoch_k'] = k.iloc[i]
        row['regime'] = regime.iloc[i]
        
        # Find relevant targets (within reasonable distance)
        relevant = [tgt for tgt in all_targets if abs(row['close'] - tgt.level) / row['close'] < 0.5]  # Relaxed
        if relevant:
            # Score and select top
            for tgt in relevant:
                tgt.probability = score_target_probability(tgt, df, row, theta)
            relevant.sort(key=lambda x: x.probability, reverse=True)
            top_support = [t for t in relevant[:3] if t.type == 'support']
            top_resist = [t for t in relevant[:3] if t.type == 'resistance']
            
            target_cols[t] = {
                'support_levels': [t.level for t in top_support],
                'support_probs': [t.probability for t in top_support],
                'resist_levels': [t.level for t in top_resist],
                'resist_probs': [t.probability for t in top_resist]
            }
        else:
            target_cols[t] = {
                'support_levels': [],
                'support_probs': [],
                'resist_levels': [],
                'resist_probs': []
            }
    
    # Convert to DataFrame
    target_df = pd.DataFrame.from_dict(target_cols, orient='index')
    return target_df