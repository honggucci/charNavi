"""
Wyckoff Phase Detection & Bidirectional Accumulation Strategies

=== 전통적 패턴 (Classic Patterns) ===

Accumulation (롱 축적) - 바닥권:
  Phase A: Selling Climax (SC) → Automatic Rally (AR) - 거미줄 기법
  Phase B: 횡보 구간 - 안전마진 전략 (박스 하단 매수)
  Phase C: Spring - 기존 전략 (잔여 자금 집중)

Distribution (숏 축적) - 고점권:
  Phase A: Buying Climax (BC) → Automatic Reaction (AR) - 거미줄 기법
  Phase B: 횡보 구간 - 안전마진 전략 (박스 상단 매도)
  Phase C: UTAD - 기존 전략 (잔여 자금 집중)

=== 추세 중 패턴 (Trend-Aligned Patterns) ===

Re-Accumulation (상승 중 매집):
  - 상승 추세 도중 횡보 박스
  - 저점이 이전보다 높은 상태 (Higher Lows)
  - 고점은 유지 또는 조금 상승
  - 거래량은 흡수형 (폭발적 증가 X)
  - 의미: 상승 추세 유지하면서 추가 물량 흡수

Re-Distribution (하락 중 분산):
  - 하락 추세 도중 횡보 박스
  - 고점이 이전보다 낮은 상태 (Lower Highs)
  - 저점은 유지 또는 조금 하락
  - 거래량은 분산형
  - 의미: 하락 추세 유지하면서 보유 물량 추가 매도

박스 피보나치 그리드:
  - 박스 저점~고점 구간을 피보나치 레벨로 분할
  - 각 레벨에서 분할 진입 (거미줄)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

from wpcn.core.types import Theta, EventSignal
from wpcn.features.indicators import atr, rsi, zscore
from wpcn.wyckoff.box import box_engine_freeze


@dataclass
class WyckoffPhase:
    """Wyckoff 페이즈 상태"""
    phase: str  # 'A', 'B', 'C', 'D', 'E', 'unknown'
    sub_phase: str  # 'SC', 'AR', 'ST', 'Spring', 'UTAD', 'BC', 'LPSY', 'ReAccum', 'ReDistrib', etc.
    direction: str  # 'accumulation', 'distribution', 're_accumulation', 're_distribution', 'unknown'
    confidence: float  # 0-1
    start_idx: int
    box_low: float
    box_high: float


@dataclass
class AccumulationSignal:
    """축적 신호 (거미줄/안전마진) - 롱/숏 양방향 지원"""
    t_signal: Any  # timestamp
    side: str  # 'long' for accumulation, 'short' for distribution
    event: str  # 'spider_web_fib382', 'spider_web_fib50', 'safety_margin', etc.
    entry_price: float
    stop_price: float
    tp_price: float  # 익절가
    position_pct: float  # 1% = 0.01
    phase: str
    meta: Optional[Dict[str, Any]] = None


# 박스 피보나치 그리드 레벨
FIB_LEVELS = {
    'fib_236': 0.236,
    'fib_382': 0.382,
    'fib_500': 0.500,
    'fib_618': 0.618,
    'fib_786': 0.786
}

# 방향별 포지션 비율 설정
DIRECTION_RATIOS = {
    'accumulation': {'long': 0.70, 'short': 0.30},        # 전통적 매집: 롱 70%, 숏 30%
    'distribution': {'long': 0.30, 'short': 0.70},        # 전통적 분산: 롱 30%, 숏 70%
    're_accumulation': {'long': 0.80, 'short': 0.20},     # 상승 중 매집: 롱 80%, 숏 20%
    're_distribution': {'long': 0.20, 'short': 0.80},     # 하락 중 분산: 롱 20%, 숏 80%
    'unknown': {'long': 0.50, 'short': 0.50}              # 50/50
}


def detect_wyckoff_phase(
    df: pd.DataFrame,
    theta: Theta,
    lookback: int = 50
) -> pd.DataFrame:
    """
    Wyckoff 페이즈를 감지합니다. (전통적 + 추세 중 패턴)

    === 전통적 패턴 ===
    Phase Detection Logic:
    - Phase A: 큰 하락 후 급반등 (SC → AR) / 큰 상승 후 급반락 (BC → AR)
    - Phase B: 횡보 구간 (박스권 내에서 가격 진동)
    - Phase C: Spring (박스 하단 돌파 후 회복) / UTAD (박스 상단 돌파 후 회귀)
    - Phase D: Markup/Markdown 시작
    - Phase E: 추세 지속

    === 추세 중 패턴 ===
    Re-Accumulation (상승 중 매집):
    - 상승 추세 도중 횡보 박스
    - Higher Lows (저점이 점점 높아짐)
    - EMA 50 > EMA 200 (상승 추세 확인)

    Re-Distribution (하락 중 분산):
    - 하락 추세 도중 횡보 박스
    - Lower Highs (고점이 점점 낮아짐)
    - EMA 50 < EMA 200 (하락 추세 확인)
    """
    _atr = atr(df, theta.atr_len)
    box = box_engine_freeze(df, theta)  # 롤링 박스 (참조용)
    _rsi = rsi(df['close'], 14)
    z = zscore(df['close'], 20)

    # EMA for trend detection (Re-Accumulation / Re-Distribution)
    ema_50 = df['close'].ewm(span=50, adjust=False).mean()
    ema_200 = df['close'].ewm(span=200, adjust=False).mean()

    df2 = df.join(_atr.rename('atr')).join(box).join(_rsi.rename('rsi')).join(z.rename('zscore'))

    # 결과 DataFrame
    n = len(df)
    phases = pd.DataFrame(index=df.index)
    phases['phase'] = 'unknown'
    phases['sub_phase'] = ''
    phases['direction'] = 'unknown'
    phases['confidence'] = 0.5
    phases['box_low'] = box['box_low']
    phases['box_high'] = box['box_high']

    # 거래량 분석
    vol_ma = df['volume'].rolling(20, min_periods=5).mean()
    vol_ratio = df['volume'] / (vol_ma + 1e-12)

    # 가격 변화
    ret = df['close'].pct_change()

    # Pre-extract arrays
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    open_arr = df['open'].values
    vol_ratio_arr = vol_ratio.values
    ret_arr = ret.values
    atr_arr = _atr.values
    rsi_arr = _rsi.values
    z_arr = z.values
    bl_arr = box['box_low'].values
    bh_arr = box['box_high'].values
    bw_arr = box['box_width'].values
    ema_50_arr = ema_50.values
    ema_200_arr = ema_200.values

    # Phase detection state
    current_phase = 'unknown'
    phase_start = 0
    direction = 'unknown'

    for i in range(lookback, n):
        # Skip if NaN
        if np.isnan(atr_arr[i]) or np.isnan(bl_arr[i]):
            continue

        close = close_arr[i]
        bl, bh = bl_arr[i], bh_arr[i]
        bw = bw_arr[i]
        at = atr_arr[i]
        curr_rsi = rsi_arr[i]
        curr_z = z_arr[i]
        vol_r = vol_ratio_arr[i]

        # === Phase A Detection: Selling/Buying Climax ===
        recent_ret = ret_arr[max(0, i-5):i+1]
        recent_vol_r = vol_ratio_arr[max(0, i-5):i+1]

        # === Accumulation: SC (Selling Climax) 감지 ===
        # 큰 음봉 + 거래량 급증 + RSI 과매도 → SC
        if (np.nanmin(recent_ret) < -0.015 and  # 1.5% 이상 하락 봉 존재
            np.nanmax(recent_vol_r) > 1.3 and  # 평균 1.3배 이상 거래량
            curr_rsi < 45):  # RSI 45 이하
            current_phase = 'A'
            direction = 'accumulation'
            phase_start = i
            phases.iloc[i, phases.columns.get_loc('phase')] = 'A'
            phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'SC'
            phases.iloc[i, phases.columns.get_loc('direction')] = 'accumulation'
            phases.iloc[i, phases.columns.get_loc('confidence')] = 0.7
            continue

        # === Distribution: BC (Buying Climax) 감지 ===
        # 큰 양봉 + 거래량 급증 + RSI 과매수 → BC
        if (np.nanmax(recent_ret) > 0.015 and  # 1.5% 이상 상승 봉 존재
            np.nanmax(recent_vol_r) > 1.3 and  # 평균 1.3배 이상 거래량
            curr_rsi > 55):  # RSI 55 이상
            current_phase = 'A'
            direction = 'distribution'
            phase_start = i
            phases.iloc[i, phases.columns.get_loc('phase')] = 'A'
            phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'BC'
            phases.iloc[i, phases.columns.get_loc('direction')] = 'distribution'
            phases.iloc[i, phases.columns.get_loc('confidence')] = 0.7
            continue

        # AR 감지: SC/BC 후 반등/반락
        if current_phase == 'A' and i - phase_start < 15:
            # Accumulation: SC 후 반등
            if direction == 'accumulation':
                if (np.nanmax(recent_ret) > 0.01 and close > bl + 0.3 * bw):
                    phases.iloc[i, phases.columns.get_loc('phase')] = 'A'
                    phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'AR'
                    phases.iloc[i, phases.columns.get_loc('direction')] = 'accumulation'
                    phases.iloc[i, phases.columns.get_loc('confidence')] = 0.7
                    continue
            # Distribution: BC 후 반락
            elif direction == 'distribution':
                if (np.nanmin(recent_ret) < -0.01 and close < bh - 0.3 * bw):
                    phases.iloc[i, phases.columns.get_loc('phase')] = 'A'
                    phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'AR_dist'  # Distribution AR
                    phases.iloc[i, phases.columns.get_loc('direction')] = 'distribution'
                    phases.iloc[i, phases.columns.get_loc('confidence')] = 0.7
                    continue

        # === Phase B Detection: 횡보 구간 ===
        # 박스권 내에서 가격 진동 (z-score가 ±2 이내로 완화)
        margin = 0.1 * bw if bw > 0 else 0
        in_box = (bl - margin) <= close <= (bh + margin)
        z_bounded = abs(curr_z) < 2.0 if not np.isnan(curr_z) else True  # 더 완화

        # === Re-Accumulation / Re-Distribution 감지 ===
        # EMA 기반 추세 확인
        ema50 = ema_50_arr[i] if not np.isnan(ema_50_arr[i]) else close
        ema200 = ema_200_arr[i] if not np.isnan(ema_200_arr[i]) else close
        is_uptrend = ema50 > ema200  # 상승 추세
        is_downtrend = ema50 < ema200  # 하락 추세

        # Higher Lows / Lower Highs 감지 (최근 20봉)
        if i >= 20:
            recent_lows = low_arr[i-20:i+1]
            recent_highs = high_arr[i-20:i+1]

            # 최근 저점들의 추세 (Higher Lows)
            first_half_low = np.min(recent_lows[:10])
            second_half_low = np.min(recent_lows[10:])
            has_higher_lows = second_half_low > first_half_low

            # 최근 고점들의 추세 (Lower Highs)
            first_half_high = np.max(recent_highs[:10])
            second_half_high = np.max(recent_highs[10:])
            has_lower_highs = second_half_high < first_half_high
        else:
            has_higher_lows = False
            has_lower_highs = False

        if in_box and z_bounded:
            # === Re-Accumulation: 상승 추세 중 횡보 + Higher Lows ===
            if is_uptrend and has_higher_lows and vol_r < 1.5:
                current_phase = 'B'
                direction = 're_accumulation'
                phases.iloc[i, phases.columns.get_loc('phase')] = 'B'
                phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'ReAccum'
                phases.iloc[i, phases.columns.get_loc('direction')] = 're_accumulation'
                phases.iloc[i, phases.columns.get_loc('confidence')] = 0.75
                continue

            # === Re-Distribution: 하락 추세 중 횡보 + Lower Highs ===
            if is_downtrend and has_lower_highs and vol_r < 1.5:
                current_phase = 'B'
                direction = 're_distribution'
                phases.iloc[i, phases.columns.get_loc('phase')] = 'B'
                phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'ReDistrib'
                phases.iloc[i, phases.columns.get_loc('direction')] = 're_distribution'
                phases.iloc[i, phases.columns.get_loc('confidence')] = 0.75
                continue

            # Phase A 후 또는 독립적 횡보 (전통적 패턴)
            if current_phase in ['A', 'B'] or (current_phase == 'unknown' and vol_r < 2.0):
                current_phase = 'B'
                phases.iloc[i, phases.columns.get_loc('phase')] = 'B'
                phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'ST'  # Secondary Test
                phases.iloc[i, phases.columns.get_loc('direction')] = direction if direction != 'unknown' else 'accumulation'
                phases.iloc[i, phases.columns.get_loc('confidence')] = 0.6
                continue

        # === Phase C Detection: Spring/UTAD ===
        # 박스 하단 아래로 돌파 → Spring 가능성
        if close < bl - 0.1 * at:
            current_phase = 'C'
            phases.iloc[i, phases.columns.get_loc('phase')] = 'C'
            phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'Spring_forming'
            phases.iloc[i, phases.columns.get_loc('direction')] = 'accumulation'
            phases.iloc[i, phases.columns.get_loc('confidence')] = 0.8
            continue

        # 박스 상단 위로 돌파 → UTAD 가능성
        if close > bh + 0.1 * at:
            current_phase = 'C'
            phases.iloc[i, phases.columns.get_loc('phase')] = 'C'
            phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'UTAD_forming'
            phases.iloc[i, phases.columns.get_loc('direction')] = 'distribution'
            phases.iloc[i, phases.columns.get_loc('confidence')] = 0.8
            continue

        # === Phase D Detection: Markup/Markdown 시작 ===
        # Spring 후 박스 상단 돌파 → Markup
        if current_phase == 'C' and close > bh:
            phases.iloc[i, phases.columns.get_loc('phase')] = 'D'
            phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'SOS'  # Sign of Strength
            phases.iloc[i, phases.columns.get_loc('direction')] = 'accumulation'
            phases.iloc[i, phases.columns.get_loc('confidence')] = 0.75
            continue

        # UTAD 후 박스 하단 돌파 → Markdown
        if current_phase == 'C' and close < bl:
            phases.iloc[i, phases.columns.get_loc('phase')] = 'D'
            phases.iloc[i, phases.columns.get_loc('sub_phase')] = 'SOW'  # Sign of Weakness
            phases.iloc[i, phases.columns.get_loc('direction')] = 'distribution'
            phases.iloc[i, phases.columns.get_loc('confidence')] = 0.75
            continue

        # === Default: 이전 상태 유지 ===
        if i > 0:
            phases.iloc[i, phases.columns.get_loc('phase')] = phases.iloc[i-1]['phase']
            phases.iloc[i, phases.columns.get_loc('sub_phase')] = phases.iloc[i-1]['sub_phase']
            phases.iloc[i, phases.columns.get_loc('direction')] = phases.iloc[i-1]['direction']
            phases.iloc[i, phases.columns.get_loc('confidence')] = max(0.3, phases.iloc[i-1]['confidence'] - 0.01)

    return phases


def detect_spider_web_signals(
    df: pd.DataFrame,
    theta: Theta,
    phases_df: pd.DataFrame,
    position_pct: float = 0.01  # 1%씩 매수/매도
) -> List[AccumulationSignal]:
    """
    박스 피보나치 그리드 거미줄 기법 (Box Fibonacci Grid Spider Web)

    박스 저점~고점을 피보나치 레벨로 분할하여 분할 진입

    Accumulation (롱):
      - 박스 하단 피보나치 레벨에서 롱 진입
      - fib_236, fib_382, fib_500 레벨에서 매수
      - 손절: 진입가 -1.0%
      - 익절: 진입가 +1.5%

    Distribution (숏):
      - 박스 상단 피보나치 레벨에서 숏 진입
      - fib_618, fib_786 레벨 및 박스 상단에서 매도
      - 손절: 진입가 +1.0%
      - 익절: 진입가 -1.5%
    """
    signals: List[AccumulationSignal] = []

    _atr = atr(df, theta.atr_len)

    # Pre-extract arrays
    idx = df.index
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr_arr = _atr.values

    # 롤링 박스 사용
    bl_arr = phases_df['box_low'].values
    bh_arr = phases_df['box_high'].values

    # 피보나치 레벨 터치 추적
    long_fib_touched = set()
    short_fib_touched = set()
    last_long_i = -10
    last_short_i = -10
    cooldown = 4

    n = len(df)

    for i in range(50, n):
        if np.isnan(atr_arr[i]) or np.isnan(bl_arr[i]):
            continue

        phase = phases_df.iloc[i]['phase']
        direction = phases_df.iloc[i]['direction']

        close = close_arr[i]
        at = atr_arr[i]
        bl = bl_arr[i]
        bh = bh_arr[i]
        bw = bh - bl

        if bw <= 0:
            continue

        # 방향별 포지션 비율 적용
        ratios = DIRECTION_RATIOS.get(direction, DIRECTION_RATIOS['unknown'])

        # === 박스 피보나치 그리드 롱 진입 (박스 하단 쪽) ===
        # fib_236, fib_382, fib_500 레벨 (박스 저점 기준)
        # 반등 확인: 현재 종가가 이전 종가보다 높아야 함 (상승 캔들)
        is_bullish_candle = i > 0 and close > close_arr[i-1]

        for fib_name, fib_pct in [('fib_236', 0.236), ('fib_382', 0.382), ('fib_500', 0.500)]:
            if fib_name in long_fib_touched:
                continue

            # 피보나치 레벨 계산 (박스 저점부터)
            fib_price = bl + bw * fib_pct

            # 가격이 피보나치 레벨 근처 + 반등 확인 (상승 캔들)
            if abs(close - fib_price) / at < 0.5 and close <= fib_price and is_bullish_candle:
                if i - last_long_i >= cooldown:
                    long_fib_touched.add(fib_name)
                    adjusted_pct = position_pct * ratios['long']
                    if adjusted_pct > 0.001:
                        # 진입가 기준 % 방식 TP/SL
                        tp_target = close * 1.015  # +1.5% 익절
                        sl_target = close * 0.990  # -1.0% 손절

                        signals.append(AccumulationSignal(
                            t_signal=idx[i],
                            side='long',
                            event=f'spider_box_{fib_name}',
                            entry_price=float(close),
                            stop_price=float(sl_target),
                            tp_price=float(tp_target),
                            position_pct=adjusted_pct,
                            phase=phase,
                            meta={
                                'fib_level': fib_pct,
                                'fib_price': fib_price,
                                'box_low': bl,
                                'box_high': bh,
                                'direction': direction,
                                'ratio': ratios['long'],
                                'atr': at
                            }
                        ))
                        last_long_i = i

        # === 박스 피보나치 그리드 숏 진입 (박스 상단 쪽) ===
        # 반락 확인: 현재 종가가 이전 종가보다 낮아야 함 (하락 캔들)
        is_bearish_candle = i > 0 and close < close_arr[i-1]

        for fib_name, fib_pct in [('fib_618', 0.618), ('fib_786', 0.786)]:
            if fib_name in short_fib_touched:
                continue

            fib_price = bl + bw * fib_pct

            # 가격이 피보나치 레벨 근처 + 반락 확인 (하락 캔들)
            if abs(close - fib_price) / at < 0.5 and close >= fib_price and is_bearish_candle:
                if i - last_short_i >= cooldown:
                    short_fib_touched.add(fib_name)
                    adjusted_pct = position_pct * ratios['short']
                    if adjusted_pct > 0.001:
                        # 진입가 기준 % 방식 TP/SL
                        tp_target = close * 0.985  # -1.5% 익절 (숏)
                        sl_target = close * 1.010  # +1.0% 손절 (숏)

                        signals.append(AccumulationSignal(
                            t_signal=idx[i],
                            side='short',
                            event=f'spider_box_{fib_name}',
                            entry_price=float(close),
                            stop_price=float(sl_target),
                            tp_price=float(tp_target),
                            position_pct=adjusted_pct,
                            phase=phase,
                            meta={
                                'fib_level': fib_pct,
                                'fib_price': fib_price,
                                'box_low': bl,
                                'box_high': bh,
                                'direction': direction,
                                'ratio': ratios['short'],
                                'atr': at
                            }
                        ))
                        last_short_i = i

        # 박스가 변경되면 터치 기록 리셋
        if i > 0 and (bl != bl_arr[i-1] or bh != bh_arr[i-1]):
            long_fib_touched = set()
            short_fib_touched = set()

    return signals


def detect_safety_margin_signals(
    df: pd.DataFrame,
    theta: Theta,
    phases_df: pd.DataFrame,
    position_pct: float = 0.01  # 1%씩 매수/매도
) -> List[AccumulationSignal]:
    """
    양방향 안전마진 전략 (Bidirectional Safety Margin Strategy)

    Phase B (횡보 구간)에서 박스 경계 터치 시 분할 진입

    롱 (Accumulation):
      - 박스 하단 근처(0.5 ATR 이내)에서 롱 진입
      - 손절: 진입가 -1.0%
      - 익절: 진입가 +1.5%

    숏 (Distribution):
      - 박스 상단 근처(0.5 ATR 이내)에서 숏 진입
      - 손절: 진입가 +1.0%
      - 익절: 진입가 -1.5%

    목표: 평균 진입가를 유리하게 하면서 Phase C 대비
    """
    signals: List[AccumulationSignal] = []

    _atr = atr(df, theta.atr_len)

    # Pre-extract arrays
    idx = df.index
    close_arr = df['close'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    atr_arr = _atr.values

    # 롤링 박스 사용
    bl_arr = phases_df['box_low'].values
    bh_arr = phases_df['box_high'].values

    last_long_i = -10
    last_short_i = -10
    cooldown = 4

    n = len(df)

    for i in range(50, n):
        if np.isnan(atr_arr[i]) or np.isnan(bl_arr[i]):
            continue

        phase = phases_df.iloc[i]['phase']
        direction = phases_df.iloc[i]['direction']

        close = close_arr[i]
        high = high_arr[i]
        low = low_arr[i]
        at = atr_arr[i]
        bl = bl_arr[i]
        bh = bh_arr[i]
        bw = bh - bl

        if bw <= 0:
            continue

        # 방향별 포지션 비율 적용
        ratios = DIRECTION_RATIOS.get(direction, DIRECTION_RATIOS['unknown'])

        # === 롱: 박스 하단 근처 ===
        # 반등 확인: 현재 종가가 이전 종가보다 높아야 함 (상승 캔들)
        is_bullish_candle = i > 0 and close > close_arr[i-1]

        box_pos = (close - bl) / bw
        if box_pos < 0.25 and i - last_long_i >= cooldown:  # 박스 하단 25% 이내
            # 박스 하단 근처 또는 터치 + 반등 캔들 확인
            distance_to_bottom = close - bl
            if (0 < distance_to_bottom < 0.5 * at or (low <= bl + 0.2 * at and close > bl)) and is_bullish_candle:
                adjusted_pct = position_pct * ratios['long']
                if adjusted_pct > 0.001:
                    # 진입가 기준 % 방식 TP/SL
                    tp_target = close * 1.015  # +1.5% 익절
                    sl_target = close * 0.990  # -1.0% 손절

                    event_name = 'safety_margin_bottom' if distance_to_bottom < 0.5 * at else 'safety_margin_touch'
                    signals.append(AccumulationSignal(
                        t_signal=idx[i],
                        side='long',
                        event=event_name,
                        entry_price=float(close),
                        stop_price=float(sl_target),
                        tp_price=float(tp_target),
                        position_pct=adjusted_pct,
                        phase='B',
                        meta={
                            'box_low': bl,
                            'box_high': bh,
                            'box_pos': box_pos,
                            'direction': direction,
                            'ratio': ratios['long'],
                            'atr': at
                        }
                    ))
                    last_long_i = i

        # === 숏: 박스 상단 근처 ===
        # 반락 확인: 현재 종가가 이전 종가보다 낮아야 함 (하락 캔들)
        is_bearish_candle = i > 0 and close < close_arr[i-1]

        if box_pos > 0.75 and i - last_short_i >= cooldown:  # 박스 상단 25% 이내
            distance_to_top = bh - close
            # 박스 상단 근처 또는 터치 + 반락 캔들 확인
            if (0 < distance_to_top < 0.5 * at or (high >= bh - 0.2 * at and close < bh)) and is_bearish_candle:
                adjusted_pct = position_pct * ratios['short']
                if adjusted_pct > 0.001:
                    # 진입가 기준 % 방식 TP/SL
                    tp_target = close * 0.985  # -1.5% 익절 (숏)
                    sl_target = close * 1.010  # +1.0% 손절 (숏)

                    event_name = 'safety_margin_top' if distance_to_top < 0.5 * at else 'safety_margin_touch_top'
                    signals.append(AccumulationSignal(
                        t_signal=idx[i],
                        side='short',
                        event=event_name,
                        entry_price=float(close),
                        stop_price=float(sl_target),
                        tp_price=float(tp_target),
                        position_pct=adjusted_pct,
                        phase='B',
                        meta={
                            'box_low': bl,
                            'box_high': bh,
                            'box_pos': box_pos,
                            'direction': direction,
                            'ratio': ratios['short'],
                            'atr': at
                        }
                    ))
                    last_short_i = i

    return signals


def detect_phase_signals(
    df: pd.DataFrame,
    theta: Theta,
    accumulation_pct: float = 0.01
) -> Tuple[pd.DataFrame, List[AccumulationSignal], List[AccumulationSignal]]:
    """
    전체 Phase 기반 신호 감지 (양방향)

    Returns:
        phases_df: 각 봉의 Wyckoff 페이즈 정보
        spider_signals: 거미줄 기법 신호 (박스 피보나치 그리드)
        safety_signals: 안전마진 신호 (박스 경계)
    """
    # 1. 페이즈 감지 (Accumulation + Distribution)
    phases_df = detect_wyckoff_phase(df, theta)

    # 페이즈 통계
    accum_count = (phases_df['direction'] == 'accumulation').sum()
    dist_count = (phases_df['direction'] == 'distribution').sum()
    reaccum_count = (phases_df['direction'] == 're_accumulation').sum()
    redistrib_count = (phases_df['direction'] == 're_distribution').sum()
    print(f"[Phase Detection] Classic Accum: {accum_count}, Classic Distrib: {dist_count}")
    print(f"[Phase Detection] Re-Accum: {reaccum_count}, Re-Distrib: {redistrib_count}")

    # 2. 박스 피보나치 그리드 거미줄 기법
    spider_signals = detect_spider_web_signals(
        df, theta, phases_df,
        position_pct=accumulation_pct
    )

    # 3. 양방향 안전마진 전략
    safety_signals = detect_safety_margin_signals(
        df, theta, phases_df,
        position_pct=accumulation_pct
    )

    # 롱/숏 통계
    spider_long = sum(1 for s in spider_signals if s.side == 'long')
    spider_short = sum(1 for s in spider_signals if s.side == 'short')
    safety_long = sum(1 for s in safety_signals if s.side == 'long')
    safety_short = sum(1 for s in safety_signals if s.side == 'short')

    print(f"[Spider Web] Long: {spider_long}, Short: {spider_short}")
    print(f"[Safety Margin] Long: {safety_long}, Short: {safety_short}")
    print(f"[Total] Long: {spider_long + safety_long}, Short: {spider_short + safety_short}")

    return phases_df, spider_signals, safety_signals
