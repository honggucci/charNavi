"""
MTF (Multi-Timeframe) Scoring System V2
Based on STRATEGY_ARCHITECTURE.md + GPT 개선안

핵심 개선:
1. Context/Trigger 점수 분리
   - Context (1W/1D/4H/1H): 레짐/방향 → 진입 허용/금지 + 사이징
   - Trigger (15M/5M/Spring/UTAD): 타이밍/이벤트 → 진입 결정

2. Phase 기반 리스크 관리
   - Phase A/B: 낮은 리스크 (0.5~1%)
   - Phase C/D: 높은 리스크 (2~2.5%)
"""
from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

from wpcn._02_data.resample import resample_ohlcv, merge_asof_back
from wpcn._03_common._02_features.indicators import rsi, adx, atr
from wpcn._03_common._03_wyckoff.phases import detect_phase_signals
from wpcn._03_common._01_core.types import Theta


# Phase별 리스크 비율 (계좌 대비 손실 허용)
PHASE_RISK_MAP = {
    'A': 0.005,           # Phase A (의심): 0.5%
    'B': 0.010,           # Phase B (형성): 1%
    'C': 0.020,           # Phase C (스프링): 2%
    'D': 0.025,           # Phase D (확정): 2.5%
    'E': 0.015,           # Phase E (추세): 1.5%
    'ACCUMULATION': 0.015,
    'RE_ACCUMULATION': 0.020,
    'DISTRIBUTION': 0.015,
    'RE_DISTRIBUTION': 0.020,
    'MARKUP': 0.020,
    'MARKDOWN': 0.020,
    'RANGING': 0.010,
    'UNKNOWN': 0.005,
}


@dataclass
class MTFScore:
    """Multi-timeframe score for a single timestamp"""
    timestamp: Any
    long_score: float = 0.0
    short_score: float = 0.0
    tf_alignment_long: int = 0  # How many timeframes agree on LONG
    tf_alignment_short: int = 0  # How many timeframes agree on SHORT

    # === Context/Trigger 분리 ===
    context_score_long: float = 0.0   # 1W/1D/4H/1H (레짐/방향)
    context_score_short: float = 0.0
    trigger_score_long: float = 0.0   # 15M/5M/Spring/UTAD (타이밍/이벤트)
    trigger_score_short: float = 0.0

    # Individual TF contributions
    tf_4h_long: float = 0.0
    tf_4h_short: float = 0.0
    tf_1h_long: float = 0.0
    tf_1h_short: float = 0.0
    tf_15m_long: float = 0.0
    tf_15m_short: float = 0.0
    tf_5m_long: float = 0.0
    tf_5m_short: float = 0.0
    tf_1d_long: float = 0.0
    tf_1d_short: float = 0.0
    tf_1w_long: float = 0.0
    tf_1w_short: float = 0.0

    # Metadata
    meta: Dict[str, Any] = None

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}


def get_phase_risk_pct(phase: str) -> float:
    """Phase에 따른 리스크 비율 반환"""
    return PHASE_RISK_MAP.get(phase, 0.01)


def calc_confidence_mult(nav_conf: float, trigger_score: float, trigger_min: float = 2.0) -> float:
    """
    Navigation confidence와 trigger score로 신뢰도 배수 계산

    Returns: 0.5 ~ 1.5 배수
    """
    # nav_conf: 0~1 → 0.5~1.5
    c = 0.5 + 1.0 * max(0.0, min(nav_conf, 1.0))

    # trigger_score가 낮으면 축소
    if trigger_score < trigger_min:
        c *= 0.8

    return max(0.5, min(c, 1.5))


def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    """Calculate EMA"""
    return close.ewm(span=period, adjust=False).mean()


def analyze_trend_ema(df: pd.DataFrame, ema_short: int = 50, ema_long: int = 200) -> pd.DataFrame:
    """
    Analyze trend using EMA 50/200 crossover

    Returns DataFrame with columns:
    - trend: 'BULLISH', 'BEARISH', 'NEUTRAL'
    - long_score: +2 if BULLISH, -2 if counter-trend
    - short_score: +2 if BEARISH, -2 if counter-trend
    """
    close = df['close']
    ema50 = calculate_ema(close, ema_short)
    ema200 = calculate_ema(close, ema_long)

    result = pd.DataFrame(index=df.index)
    result['ema50'] = ema50
    result['ema200'] = ema200

    # Price > EMA50 > EMA200 → BULLISH
    # Price < EMA50 < EMA200 → BEARISH
    bullish = (close > ema50) & (ema50 > ema200)
    bearish = (close < ema50) & (ema50 < ema200)

    result['trend'] = 'NEUTRAL'
    result.loc[bullish, 'trend'] = 'BULLISH'
    result.loc[bearish, 'trend'] = 'BEARISH'

    result['long_score'] = 0.0
    result['short_score'] = 0.0

    # BULLISH → +2 for LONG, -2 for SHORT (counter-trend penalty)
    result.loc[bullish, 'long_score'] = 2.0
    result.loc[bullish, 'short_score'] = -2.0

    # BEARISH → +2 for SHORT, -2 for LONG (counter-trend penalty)
    result.loc[bearish, 'short_score'] = 2.0
    result.loc[bearish, 'long_score'] = -2.0

    return result


def analyze_wyckoff_structure(df: pd.DataFrame, theta: Theta) -> pd.DataFrame:
    """
    Analyze Wyckoff structure on 1H

    Returns DataFrame with columns:
    - phase: 'ACCUMULATION', 'RE_ACCUMULATION', 'DISTRIBUTION', 'RE_DISTRIBUTION', 'MARKUP', 'MARKDOWN', 'RANGING'
    - spring_detected: bool
    - utad_detected: bool
    - long_score: Points for LONG
    - short_score: Points for SHORT
    """
    # Use existing phase detection
    phases_df, accum_signals, distrib_signals = detect_phase_signals(df, theta, accumulation_pct=0.03)

    result = pd.DataFrame(index=df.index)
    result['phase'] = 'RANGING'
    result['spring_detected'] = False
    result['utad_detected'] = False
    result['long_score'] = 0.0
    result['short_score'] = 0.0

    # Merge phase info
    if 'phase' in phases_df.columns:
        result['phase'] = phases_df['phase'].fillna('RANGING')

    # Calculate close position relative to box
    if 'box_high' in phases_df.columns and 'box_low' in phases_df.columns:
        box_high = phases_df['box_high']
        box_low = phases_df['box_low']
        box_mid = (box_high + box_low) / 2
        close_pos = (df['close'] - box_low) / (box_high - box_low + 1e-10)

        # Detect Spring: price breaks below box_low, then recovers above box_low
        spring = (df['low'] < box_low) & (df['close'] > box_low)
        result.loc[spring, 'spring_detected'] = True

        # Detect UTAD: price breaks above box_high, then fails below box_high
        utad = (df['high'] > box_high) & (df['close'] < box_high)
        result.loc[utad, 'utad_detected'] = True

    # Detect trend for Re-Accumulation / Re-Distribution
    close = df['close']
    ema50 = calculate_ema(close, 50)
    in_uptrend = close > ema50
    in_downtrend = close < ema50

    # Higher Lows detection (for RE_ACCUMULATION)
    lows = df['low']
    prev_low = lows.shift(1)
    higher_lows = lows > prev_low

    # Lower Highs detection (for RE_DISTRIBUTION)
    highs = df['high']
    prev_high = highs.shift(1)
    lower_highs = highs < prev_high

    # Assign scores based on phase
    for idx in result.index:
        phase = result.loc[idx, 'phase']
        spring_det = result.loc[idx, 'spring_detected']
        utad_det = result.loc[idx, 'utad_detected']

        # Traditional patterns
        if phase == 'ACCUMULATION':
            result.loc[idx, 'long_score'] += 2.0
        elif phase == 'DISTRIBUTION':
            result.loc[idx, 'short_score'] += 2.0
        elif phase == 'MARKUP':
            result.loc[idx, 'long_score'] += 1.0
        elif phase == 'MARKDOWN':
            result.loc[idx, 'short_score'] += 1.0

        # Trend-based patterns (higher scores)
        if phase == 'RANGING' or phase == 'ACCUMULATION':
            if in_uptrend.loc[idx] and higher_lows.loc[idx]:
                # RE_ACCUMULATION: resting in uptrend
                result.loc[idx, 'long_score'] += 2.5
                result.loc[idx, 'phase'] = 'RE_ACCUMULATION'

        if phase == 'RANGING' or phase == 'DISTRIBUTION':
            if in_downtrend.loc[idx] and lower_highs.loc[idx]:
                # RE_DISTRIBUTION: resting in downtrend
                result.loc[idx, 'short_score'] += 2.5
                result.loc[idx, 'phase'] = 'RE_DISTRIBUTION'

        # Spring/UTAD bonus
        if spring_det:
            result.loc[idx, 'long_score'] += 1.5
        if utad_det:
            result.loc[idx, 'short_score'] += 1.5

    return result


def detect_rsi_divergence(df: pd.DataFrame, rsi_period: int = 14, lookback: int = 20) -> pd.DataFrame:
    """
    Detect RSI divergence on 15M

    Returns DataFrame with columns:
    - regular_bullish: strength (0-1)
    - regular_bearish: strength (0-1)
    - hidden_bullish: strength (0-1)
    - hidden_bearish: strength (0-1)
    - long_score: Points for LONG
    - short_score: Points for SHORT
    """
    result = pd.DataFrame(index=df.index)
    result['regular_bullish'] = 0.0
    result['regular_bearish'] = 0.0
    result['hidden_bullish'] = 0.0
    result['hidden_bearish'] = 0.0
    result['long_score'] = 0.0
    result['short_score'] = 0.0

    rsi_vals = rsi(df['close'], rsi_period)

    # Find pivot points for price and RSI
    for i in range(lookback * 2, len(df)):
        # Get recent window
        price_window = df['close'].iloc[i-lookback*2:i+1]
        rsi_window = rsi_vals.iloc[i-lookback*2:i+1]

        # Find last two lows
        price_lows_idx = price_window.nsmallest(2).index
        if len(price_lows_idx) < 2:
            continue

        price_low1 = price_window[price_lows_idx[0]]
        price_low2 = price_window[price_lows_idx[1]]
        rsi_low1 = rsi_window[price_lows_idx[0]]
        rsi_low2 = rsi_window[price_lows_idx[1]]

        # Regular Bullish: Price LL + RSI HL
        if price_low2 < price_low1 and rsi_low2 > rsi_low1:
            strength = min(1.0, (rsi_low2 - rsi_low1) / 10.0)  # Normalize
            result.loc[df.index[i], 'regular_bullish'] = strength
            result.loc[df.index[i], 'long_score'] += 2.0 * strength

        # Hidden Bullish: Price HL + RSI LL
        if price_low2 > price_low1 and rsi_low2 < rsi_low1:
            strength = min(1.0, abs(price_low2 - price_low1) / price_low1 * 100)
            result.loc[df.index[i], 'hidden_bullish'] = strength
            result.loc[df.index[i], 'long_score'] += 1.6 * strength

        # Find last two highs
        price_highs_idx = price_window.nlargest(2).index
        if len(price_highs_idx) < 2:
            continue

        price_high1 = price_window[price_highs_idx[0]]
        price_high2 = price_window[price_highs_idx[1]]
        rsi_high1 = rsi_window[price_highs_idx[0]]
        rsi_high2 = rsi_window[price_highs_idx[1]]

        # Regular Bearish: Price HH + RSI LH
        if price_high2 > price_high1 and rsi_high2 < rsi_high1:
            strength = min(1.0, (rsi_high1 - rsi_high2) / 10.0)
            result.loc[df.index[i], 'regular_bearish'] = strength
            result.loc[df.index[i], 'short_score'] += 2.0 * strength

        # Hidden Bearish: Price LH + RSI HH
        if price_high2 < price_high1 and rsi_high2 > rsi_high1:
            strength = min(1.0, abs(price_high2 - price_high1) / price_high1 * 100)
            result.loc[df.index[i], 'hidden_bearish'] = strength
            result.loc[df.index[i], 'short_score'] += 1.6 * strength

    return result


def analyze_5m_timing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze 5M entry timing

    NOTE: Original docs mentioned Stoch RSI, but user corrected this.
    Using price action and reversal candles instead.

    Returns DataFrame with columns:
    - oversold: bool
    - overbought: bool
    - reversal_long: bool (hammer, bullish engulfing)
    - reversal_short: bool (shooting star, bearish engulfing)
    - long_score: Points for LONG
    - short_score: Points for SHORT
    """
    result = pd.DataFrame(index=df.index)
    result['oversold'] = False
    result['overbought'] = False
    result['reversal_long'] = False
    result['reversal_short'] = False
    result['long_score'] = 0.0
    result['short_score'] = 0.0

    # Use RSI for oversold/overbought
    rsi_vals = rsi(df['close'], 14)
    result['oversold'] = rsi_vals < 30
    result['overbought'] = rsi_vals > 70

    # Reversal candle patterns
    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']

    body = abs(c - o)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    # Hammer pattern (bullish): small body, long lower wick, little upper wick
    hammer = (lower_wick > 2 * body) & (upper_wick < body * 0.3) & (c > o)
    result.loc[hammer, 'reversal_long'] = True

    # Shooting star (bearish): small body, long upper wick, little lower wick
    shooting_star = (upper_wick > 2 * body) & (lower_wick < body * 0.3) & (c < o)
    result.loc[shooting_star, 'reversal_short'] = True

    # Bullish engulfing
    prev_o = o.shift(1)
    prev_c = c.shift(1)
    bullish_engulf = (prev_c < prev_o) & (c > o) & (o < prev_c) & (c > prev_o)
    result.loc[bullish_engulf, 'reversal_long'] = True

    # Bearish engulfing
    bearish_engulf = (prev_c > prev_o) & (c < o) & (o > prev_c) & (c < prev_o)
    result.loc[bearish_engulf, 'reversal_short'] = True

    # Scoring
    result.loc[result['oversold'], 'long_score'] += 1.0
    result.loc[result['overbought'], 'short_score'] += 1.0
    result.loc[result['reversal_long'], 'long_score'] += 1.5
    result.loc[result['reversal_short'], 'short_score'] += 1.5

    return result


def compute_mtf_scores(
    df: pd.DataFrame,
    theta: Theta,
    timeframes: Optional[list] = None
) -> pd.DataFrame:
    """
    Compute multi-timeframe scores for each bar in df

    Args:
        df: 5M OHLCV data (base timeframe)
        theta: Wyckoff detection parameters
        timeframes: List of higher timeframes to analyze (e.g., ['15m', '1h', '4h', '1d', '1w'])

    Returns:
        DataFrame with MTF scores indexed by df.index, with columns:
        - long_score: Total LONG score
        - short_score: Total SHORT score
        - tf_alignment_long: Number of timeframes agreeing on LONG
        - tf_alignment_short: Number of timeframes agreeing on SHORT
        - tf_*_long/short: Individual timeframe contributions
        - meta_*: Metadata fields
    """
    if timeframes is None:
        timeframes = ['15m', '1h', '4h', '1d', '1w']

    # Initialize result DataFrame
    result = pd.DataFrame(index=df.index)
    result['long_score'] = 0.0
    result['short_score'] = 0.0
    result['tf_alignment_long'] = 0
    result['tf_alignment_short'] = 0

    # === Context/Trigger 분리 초기화 ===
    result['context_score_long'] = 0.0   # 1W/1D/4H/1H
    result['context_score_short'] = 0.0
    result['trigger_score_long'] = 0.0   # 15M/5M/Spring/UTAD
    result['trigger_score_short'] = 0.0

    # 5M timing (base timeframe - already at 5M) → TRIGGER
    print(f"  [MTF] Analyzing 5M timing...")
    timing_5m = analyze_5m_timing(df)
    result['tf_5m_long'] = timing_5m['long_score']
    result['tf_5m_short'] = timing_5m['short_score']
    result['long_score'] += timing_5m['long_score']
    result['short_score'] += timing_5m['short_score']
    result['trigger_score_long'] += timing_5m['long_score']  # Trigger에 추가
    result['trigger_score_short'] += timing_5m['short_score']
    result['tf_alignment_long'] += (timing_5m['long_score'] > 0).astype(int)
    result['tf_alignment_short'] += (timing_5m['short_score'] > 0).astype(int)

    # Higher timeframes
    for tf in timeframes:
        print(f"  [MTF] Analyzing {tf}...")

        # Resample to higher timeframe
        df_tf = resample_ohlcv(df, tf)

        if tf == '15m':
            # 15M: RSI Divergence → TRIGGER
            div_15m = detect_rsi_divergence(df_tf, rsi_period=14, lookback=20)
            div_merged = merge_asof_back(df.index, div_15m[['long_score', 'short_score']])
            result['tf_15m_long'] = div_merged['long_score'].fillna(0)
            result['tf_15m_short'] = div_merged['short_score'].fillna(0)
            result['long_score'] += result['tf_15m_long']
            result['short_score'] += result['tf_15m_short']
            result['trigger_score_long'] += result['tf_15m_long']  # Trigger
            result['trigger_score_short'] += result['tf_15m_short']
            result['tf_alignment_long'] += (result['tf_15m_long'] > 0).astype(int)
            result['tf_alignment_short'] += (result['tf_15m_short'] > 0).astype(int)

        elif tf == '1h':
            # 1H: Wyckoff Structure → CONTEXT (Phase/구조)
            wyckoff_1h = analyze_wyckoff_structure(df_tf, theta)
            wyckoff_merged = merge_asof_back(df.index, wyckoff_1h[['long_score', 'short_score', 'phase', 'spring_detected', 'utad_detected']])
            result['tf_1h_long'] = wyckoff_merged['long_score'].fillna(0)
            result['tf_1h_short'] = wyckoff_merged['short_score'].fillna(0)
            result['meta_1h_phase'] = wyckoff_merged['phase'].fillna('RANGING')
            result['meta_spring_detected'] = wyckoff_merged['spring_detected'].fillna(False)
            result['meta_utad_detected'] = wyckoff_merged['utad_detected'].fillna(False)
            result['long_score'] += result['tf_1h_long']
            result['short_score'] += result['tf_1h_short']
            result['context_score_long'] += result['tf_1h_long']  # Context
            result['context_score_short'] += result['tf_1h_short']
            # Spring/UTAD는 Trigger에도 추가 (이벤트 신호)
            result['trigger_score_long'] += wyckoff_merged['spring_detected'].fillna(False).astype(float) * 1.5
            result['trigger_score_short'] += wyckoff_merged['utad_detected'].fillna(False).astype(float) * 1.5
            result['tf_alignment_long'] += (result['tf_1h_long'] > 0).astype(int)
            result['tf_alignment_short'] += (result['tf_1h_short'] > 0).astype(int)

        elif tf == '4h':
            # 4H: Trend Filter → CONTEXT
            trend_4h = analyze_trend_ema(df_tf, ema_short=50, ema_long=200)
            trend_merged = merge_asof_back(df.index, trend_4h[['long_score', 'short_score', 'trend']])
            result['tf_4h_long'] = trend_merged['long_score'].fillna(0)
            result['tf_4h_short'] = trend_merged['short_score'].fillna(0)
            result['meta_4h_trend'] = trend_merged['trend'].fillna('NEUTRAL')
            result['long_score'] += result['tf_4h_long']
            result['short_score'] += result['tf_4h_short']
            result['context_score_long'] += result['tf_4h_long']  # Context
            result['context_score_short'] += result['tf_4h_short']
            result['tf_alignment_long'] += (result['tf_4h_long'] > 0).astype(int)
            result['tf_alignment_short'] += (result['tf_4h_short'] > 0).astype(int)

        elif tf == '1d':
            # 1D: Daily Trend → CONTEXT
            trend_1d = analyze_trend_ema(df_tf, ema_short=50, ema_long=200)
            trend_merged = merge_asof_back(df.index, trend_1d[['long_score', 'short_score', 'trend']])
            result['tf_1d_long'] = trend_merged['long_score'].fillna(0)
            result['tf_1d_short'] = trend_merged['short_score'].fillna(0)
            result['meta_1d_trend'] = trend_merged['trend'].fillna('NEUTRAL')
            result['long_score'] += result['tf_1d_long']
            result['short_score'] += result['tf_1d_short']
            result['context_score_long'] += result['tf_1d_long']  # Context
            result['context_score_short'] += result['tf_1d_short']
            result['tf_alignment_long'] += (result['tf_1d_long'] > 0).astype(int)
            result['tf_alignment_short'] += (result['tf_1d_short'] > 0).astype(int)

        elif tf == '1w':
            # 1W: Weekly Trend → CONTEXT
            trend_1w = analyze_trend_ema(df_tf, ema_short=50, ema_long=200)
            trend_merged = merge_asof_back(df.index, trend_1w[['long_score', 'short_score', 'trend']])
            result['tf_1w_long'] = trend_merged['long_score'].fillna(0)
            result['tf_1w_short'] = trend_merged['short_score'].fillna(0)
            result['meta_1w_trend'] = trend_merged['trend'].fillna('NEUTRAL')
            result['long_score'] += result['tf_1w_long']
            result['short_score'] += result['tf_1w_short']
            result['context_score_long'] += result['tf_1w_long']  # Context
            result['context_score_short'] += result['tf_1w_short']
            result['tf_alignment_long'] += (result['tf_1w_long'] > 0).astype(int)
            result['tf_alignment_short'] += (result['tf_1w_short'] > 0).astype(int)

    print(f"  [MTF] Scoring complete.")
    print(f"    Max LONG: total={result['long_score'].max():.2f}, context={result['context_score_long'].max():.2f}, trigger={result['trigger_score_long'].max():.2f}")
    print(f"    Max SHORT: total={result['short_score'].max():.2f}, context={result['context_score_short'].max():.2f}, trigger={result['trigger_score_short'].max():.2f}")

    return result


def generate_mtf_signals(
    mtf_scores: pd.DataFrame,
    df: pd.DataFrame,
    min_score: float = 4.0,
    min_tf_alignment: int = 2,
    min_rr_ratio: float = 1.5,
    atr_period: int = 14,
    sl_atr_mult: float = 2.0,
    tp_atr_mult: float = 4.0,
    min_context_score: float = 2.0,
    min_trigger_score: float = 1.5
) -> pd.DataFrame:
    """
    Generate trading signals from MTF scores (V2: Context/Trigger 분리)

    V2 개선사항:
    - Context Score (1W/1D/4H/1H): 레짐/방향 판단 → 진입 허용/금지
    - Trigger Score (15M/5M/Spring/UTAD): 타이밍/이벤트 → 진입 결정

    Args:
        min_context_score: 최소 Context 점수 (기본 2.0) - 레짐이 올바른지
        min_trigger_score: 최소 Trigger 점수 (기본 1.5) - 타이밍이 맞는지

    Returns DataFrame with columns:
    - signal: 'LONG', 'SHORT', or None
    - entry_price: Entry price
    - stop_loss: SL price
    - take_profit: TP price
    - score: Total score
    - context_score: Context score (1W/1D/4H/1H)
    - trigger_score: Trigger score (15M/5M/Spring/UTAD)
    - rr_ratio: Risk/Reward ratio
    - phase: Wyckoff phase (for risk sizing)
    - risk_pct: Recommended risk % based on phase
    """
    result = pd.DataFrame(index=mtf_scores.index)
    result['signal'] = None
    result['entry_price'] = np.nan
    result['stop_loss'] = np.nan
    result['take_profit'] = np.nan
    result['score'] = 0.0
    result['context_score'] = 0.0
    result['trigger_score'] = 0.0
    result['rr_ratio'] = 0.0
    result['phase'] = 'UNKNOWN'
    result['risk_pct'] = 0.01  # 기본 1%

    # Calculate ATR for TP/SL
    atr_vals = atr(df, atr_period)

    for i in range(len(mtf_scores)):
        idx = mtf_scores.index[i]

        # === Context/Trigger 점수 읽기 ===
        context_long = mtf_scores.loc[idx, 'context_score_long']
        context_short = mtf_scores.loc[idx, 'context_score_short']
        trigger_long = mtf_scores.loc[idx, 'trigger_score_long']
        trigger_short = mtf_scores.loc[idx, 'trigger_score_short']

        long_score = mtf_scores.loc[idx, 'long_score']
        short_score = mtf_scores.loc[idx, 'short_score']
        tf_align_long = mtf_scores.loc[idx, 'tf_alignment_long']
        tf_align_short = mtf_scores.loc[idx, 'tf_alignment_short']

        # Phase 정보 (리스크 사이징용)
        phase = mtf_scores.loc[idx, 'meta_1h_phase'] if 'meta_1h_phase' in mtf_scores.columns else 'UNKNOWN'

        entry_px = df.loc[idx, 'close']
        atr_val = atr_vals.loc[idx]

        # === V2: Context/Trigger 기반 LONG 신호 ===
        # 조건: context_ok AND trigger_ok AND tf_alignment AND RR >= min_rr
        context_ok_long = context_long >= min_context_score
        trigger_ok_long = trigger_long >= min_trigger_score
        tf_ok_long = tf_align_long >= min_tf_alignment

        if context_ok_long and trigger_ok_long and tf_ok_long:
            sl = entry_px - (atr_val * sl_atr_mult)
            tp = entry_px + (atr_val * tp_atr_mult)
            risk = entry_px - sl
            reward = tp - entry_px
            rr = reward / risk if risk > 0 else 0

            if rr >= min_rr_ratio:
                result.loc[idx, 'signal'] = 'LONG'
                result.loc[idx, 'entry_price'] = entry_px
                result.loc[idx, 'stop_loss'] = sl
                result.loc[idx, 'take_profit'] = tp
                result.loc[idx, 'score'] = long_score
                result.loc[idx, 'context_score'] = context_long
                result.loc[idx, 'trigger_score'] = trigger_long
                result.loc[idx, 'rr_ratio'] = rr
                result.loc[idx, 'phase'] = phase
                result.loc[idx, 'risk_pct'] = get_phase_risk_pct(phase)

        # === V2: Context/Trigger 기반 SHORT 신호 ===
        context_ok_short = context_short >= min_context_score
        trigger_ok_short = trigger_short >= min_trigger_score
        tf_ok_short = tf_align_short >= min_tf_alignment

        if context_ok_short and trigger_ok_short and tf_ok_short:
            sl = entry_px + (atr_val * sl_atr_mult)
            tp = entry_px - (atr_val * tp_atr_mult)
            risk = sl - entry_px
            reward = entry_px - tp
            rr = reward / risk if risk > 0 else 0

            if rr >= min_rr_ratio:
                result.loc[idx, 'signal'] = 'SHORT'
                result.loc[idx, 'entry_price'] = entry_px
                result.loc[idx, 'stop_loss'] = sl
                result.loc[idx, 'take_profit'] = tp
                result.loc[idx, 'score'] = short_score
                result.loc[idx, 'context_score'] = context_short
                result.loc[idx, 'trigger_score'] = trigger_short
                result.loc[idx, 'rr_ratio'] = rr
                result.loc[idx, 'phase'] = phase
                result.loc[idx, 'risk_pct'] = get_phase_risk_pct(phase)

    # 통계 출력
    signals = result[result['signal'].notna()]
    if len(signals) > 0:
        print(f"  [MTF Signals V2] Generated {len(signals)} signals")
        print(f"    Avg Context: {signals['context_score'].mean():.2f}, Avg Trigger: {signals['trigger_score'].mean():.2f}")
        print(f"    Avg Risk%: {signals['risk_pct'].mean()*100:.2f}%")

    return result
