"""
차트 네비게이터 V2
Chart Navigator V2

핵심 구조:
1. 와이코프 패턴 = 구조의 뼈대 (매집/분산 박스 감지)
2. Stoch RSI = 다이버전스 필터 (과매수/과매도 구간에서만)
3. RSI 다이버전스 = 스나이핑 타이밍
4. 지그재그 = 확률적 선행지표로 변환
5. MTF 피보나치 컨플루언스 = 타겟/손절 레벨

추가 견고성:
- 볼륨 프로파일 기반 지지/저항
- ATR 기반 동적 TP/SL
- 상위 타임프레임 추세 확인
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class WyckoffStructure:
    """와이코프 구조 분석 결과"""
    phase: str              # 'accumulation', 'distribution', 'markup', 'markdown', 'ranging'
    box_high: float         # 박스 상단
    box_low: float          # 박스 하단
    box_mid: float          # 박스 중앙
    volume_profile: str     # 'increasing', 'decreasing', 'neutral'
    spring_detected: bool   # 스프링 감지 (박스 하단 돌파 후 복귀)
    upthrust_detected: bool # 업쓰러스트 감지 (박스 상단 돌파 후 복귀)
    confidence: float       # 0~1


@dataclass
class ZigzagPrediction:
    """지그재그 예측 결과"""
    next_swing_direction: str   # 'up' or 'down'
    target_price: float         # 예상 타겟 가격
    probability: float          # 도달 확률 (0~1)
    expected_bars: int          # 예상 소요 봉 수
    fib_level: float           # 타겟 피보나치 레벨
    confidence: float          # 신뢰도


@dataclass
class NavigatorSignal:
    """네비게이터 매매 신호"""
    action: str             # 'LONG', 'SHORT', 'WAIT'
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    confidence: float
    reasons: List[str]      # 진입 근거


class ChartNavigatorV2:
    """
    차트 네비게이터 V2

    와이코프 구조 + Stoch RSI 필터 + RSI 다이버전스 + 지그재그 예측
    """

    def __init__(
        self,
        # RSI 설정
        rsi_period: int = 14,
        stoch_rsi_period: int = 14,
        stoch_k: int = 3,
        stoch_d: int = 3,
        # 와이코프 설정
        wyckoff_lookback: int = 50,
        box_threshold: float = 0.03,  # 3% 범위 내 횡보 = 박스
        # 지그재그 설정
        zigzag_threshold: float = 0.02,  # 2% 이상 움직임
        # 피보나치
        fib_levels: List[float] = None,
        # ATR
        atr_period: int = 14,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 3.0,
    ):
        self.rsi_period = rsi_period
        self.stoch_rsi_period = stoch_rsi_period
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.wyckoff_lookback = wyckoff_lookback
        self.box_threshold = box_threshold
        self.zigzag_threshold = zigzag_threshold
        self.fib_levels = fib_levels or [0.382, 0.5, 0.618, 0.786]
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult

    # =========================================================================
    # 1. 기본 지표 계산
    # =========================================================================

    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """RSI 계산"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calculate_stoch_rsi(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Stochastic RSI 계산"""
        rsi = self._calculate_rsi(close)

        # Stochastic of RSI
        rsi_min = rsi.rolling(self.stoch_rsi_period).min()
        rsi_max = rsi.rolling(self.stoch_rsi_period).max()

        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100

        # K와 D 라인
        k = stoch_rsi.rolling(self.stoch_k).mean()
        d = k.rolling(self.stoch_d).mean()

        return k, d

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """ATR 계산"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    # =========================================================================
    # 2. Stoch RSI 필터링된 RSI 다이버전스 감지
    # =========================================================================

    def _find_stoch_filtered_divergences(
        self,
        df: pd.DataFrame,
        lookback: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Stoch RSI가 과매수/과매도일 때만 RSI 다이버전스 감지

        - Stoch RSI < 20: 과매도 구간 → Bullish 다이버전스 체크
        - Stoch RSI > 80: 과매수 구간 → Bearish 다이버전스 체크
        """
        close = df['close']
        rsi = self._calculate_rsi(close)
        stoch_k, stoch_d = self._calculate_stoch_rsi(close)

        divergences = []

        # 피벗 포인트 찾기
        window = 5
        pivots_high = []
        pivots_low = []

        for i in range(window, len(df) - window):
            # Stoch RSI 조건 체크
            stoch_val = stoch_k.iloc[i]

            # 과매도 구간 (Stoch < 30) - Bullish 다이버전스만 체크
            if stoch_val < 30:
                if close.iloc[i] == close.iloc[i-window:i+window+1].min():
                    pivots_low.append({
                        'idx': i,
                        'price': close.iloc[i],
                        'rsi': rsi.iloc[i],
                        'stoch': stoch_val
                    })

            # 과매수 구간 (Stoch > 70) - Bearish 다이버전스만 체크
            if stoch_val > 70:
                if close.iloc[i] == close.iloc[i-window:i+window+1].max():
                    pivots_high.append({
                        'idx': i,
                        'price': close.iloc[i],
                        'rsi': rsi.iloc[i],
                        'stoch': stoch_val
                    })

        # Bullish Divergence (Regular): 가격 Lower Low + RSI Higher Low
        recent_lows = [p for p in pivots_low if p['idx'] > len(df) - lookback]
        if len(recent_lows) >= 2:
            prev = recent_lows[-2]
            curr = recent_lows[-1]

            # Regular Bullish: 가격 LL, RSI HL
            if curr['price'] < prev['price'] and curr['rsi'] > prev['rsi']:
                strength = (curr['rsi'] - prev['rsi']) / 30
                divergences.append({
                    'type': 'bullish_regular',
                    'idx': curr['idx'],
                    'price': curr['price'],
                    'rsi': curr['rsi'],
                    'stoch': curr['stoch'],
                    'strength': min(1.0, abs(strength))
                })

            # Hidden Bullish: 가격 HL, RSI LL (추세 지속)
            if curr['price'] > prev['price'] and curr['rsi'] < prev['rsi']:
                strength = (prev['rsi'] - curr['rsi']) / 30
                divergences.append({
                    'type': 'bullish_hidden',
                    'idx': curr['idx'],
                    'price': curr['price'],
                    'rsi': curr['rsi'],
                    'stoch': curr['stoch'],
                    'strength': min(1.0, abs(strength)) * 0.8  # 히든은 약간 낮은 강도
                })

        # Bearish Divergence (Regular): 가격 Higher High + RSI Lower High
        recent_highs = [p for p in pivots_high if p['idx'] > len(df) - lookback]
        if len(recent_highs) >= 2:
            prev = recent_highs[-2]
            curr = recent_highs[-1]

            # Regular Bearish: 가격 HH, RSI LH
            if curr['price'] > prev['price'] and curr['rsi'] < prev['rsi']:
                strength = (prev['rsi'] - curr['rsi']) / 30
                divergences.append({
                    'type': 'bearish_regular',
                    'idx': curr['idx'],
                    'price': curr['price'],
                    'rsi': curr['rsi'],
                    'stoch': curr['stoch'],
                    'strength': min(1.0, abs(strength))
                })

            # Hidden Bearish: 가격 LH, RSI HH (추세 지속)
            if curr['price'] < prev['price'] and curr['rsi'] > prev['rsi']:
                strength = (curr['rsi'] - prev['rsi']) / 30
                divergences.append({
                    'type': 'bearish_hidden',
                    'idx': curr['idx'],
                    'price': curr['price'],
                    'rsi': curr['rsi'],
                    'stoch': curr['stoch'],
                    'strength': min(1.0, abs(strength)) * 0.8
                })

        return divergences

    # =========================================================================
    # 3. 와이코프 구조 분석
    # =========================================================================

    def _analyze_wyckoff_structure(self, df: pd.DataFrame) -> WyckoffStructure:
        """
        와이코프 구조 분석

        - 박스 범위 감지 (매집/분산 구간)
        - 스프링/업쓰러스트 감지
        - 볼륨 프로파일 분석
        """
        lookback = min(self.wyckoff_lookback, len(df) - 1)
        recent = df.iloc[-lookback:]

        high = recent['high'].max()
        low = recent['low'].min()
        close = df['close'].iloc[-1]

        # 박스 범위 계산
        price_range = (high - low) / low
        box_mid = (high + low) / 2

        # 현재 가격 위치
        price_position = (close - low) / (high - low + 1e-10)

        # 볼륨 분석
        if 'volume' in df.columns:
            vol_recent = recent['volume'].iloc[-10:].mean()
            vol_earlier = recent['volume'].iloc[:-10].mean()

            if vol_recent > vol_earlier * 1.2:
                volume_profile = 'increasing'
            elif vol_recent < vol_earlier * 0.8:
                volume_profile = 'decreasing'
            else:
                volume_profile = 'neutral'
        else:
            volume_profile = 'neutral'

        # 스프링 감지: 박스 하단 돌파 후 빠르게 복귀
        spring_detected = False
        for i in range(-10, 0):
            if i < -len(recent):
                continue
            if recent['low'].iloc[i] < low * 0.995:  # 하단 이탈
                if recent['close'].iloc[i:].max() > box_mid:  # 복귀
                    spring_detected = True
                    break

        # 업쓰러스트 감지: 박스 상단 돌파 후 빠르게 복귀
        upthrust_detected = False
        for i in range(-10, 0):
            if i < -len(recent):
                continue
            if recent['high'].iloc[i] > high * 1.005:  # 상단 이탈
                if recent['close'].iloc[i:].min() < box_mid:  # 복귀
                    upthrust_detected = True
                    break

        # 페이즈 판정
        if price_range < self.box_threshold:
            # 좁은 범위 = 횡보 (매집 또는 분산)
            if volume_profile == 'increasing' and price_position < 0.5:
                phase = 'accumulation'
                confidence = 0.6 + (0.2 if spring_detected else 0)
            elif volume_profile == 'increasing' and price_position > 0.5:
                phase = 'distribution'
                confidence = 0.6 + (0.2 if upthrust_detected else 0)
            else:
                phase = 'ranging'
                confidence = 0.5
        else:
            # 넓은 범위 = 추세
            if price_position > 0.7:
                phase = 'markup'
                confidence = 0.7
            elif price_position < 0.3:
                phase = 'markdown'
                confidence = 0.7
            else:
                phase = 'ranging'
                confidence = 0.5

        return WyckoffStructure(
            phase=phase,
            box_high=high,
            box_low=low,
            box_mid=box_mid,
            volume_profile=volume_profile,
            spring_detected=spring_detected,
            upthrust_detected=upthrust_detected,
            confidence=confidence
        )

    # =========================================================================
    # 4. 지그재그 확률적 선행 예측
    # =========================================================================

    def _calculate_zigzag(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """지그재그 피벗 계산"""
        pivots = []
        threshold = self.zigzag_threshold

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        if len(close) < 10:
            return pivots

        # 첫 피벗 설정
        last_pivot_type = None
        last_pivot_price = close[0]
        last_pivot_idx = 0

        for i in range(1, len(close)):
            current_high = high[i]
            current_low = low[i]

            if last_pivot_type is None or last_pivot_type == 'low':
                # 고점 찾기
                change = (current_high - last_pivot_price) / last_pivot_price
                if change >= threshold:
                    if pivots and pivots[-1]['type'] == 'high':
                        # 더 높은 고점으로 업데이트
                        if current_high > pivots[-1]['price']:
                            pivots[-1] = {'idx': i, 'price': current_high, 'type': 'high'}
                    else:
                        pivots.append({'idx': i, 'price': current_high, 'type': 'high'})
                    last_pivot_type = 'high'
                    last_pivot_price = current_high
                    last_pivot_idx = i

            if last_pivot_type is None or last_pivot_type == 'high':
                # 저점 찾기
                change = (last_pivot_price - current_low) / last_pivot_price
                if change >= threshold:
                    if pivots and pivots[-1]['type'] == 'low':
                        # 더 낮은 저점으로 업데이트
                        if current_low < pivots[-1]['price']:
                            pivots[-1] = {'idx': i, 'price': current_low, 'type': 'low'}
                    else:
                        pivots.append({'idx': i, 'price': current_low, 'type': 'low'})
                    last_pivot_type = 'low'
                    last_pivot_price = current_low
                    last_pivot_idx = i

        return pivots

    def _predict_next_zigzag(
        self,
        df: pd.DataFrame,
        wyckoff: WyckoffStructure,
        divergences: List[Dict]
    ) -> Optional[ZigzagPrediction]:
        """
        다음 지그재그 스윙 포인트 예측

        와이코프 구조 + 다이버전스 + 피보나치 기반
        """
        pivots = self._calculate_zigzag(df)

        if len(pivots) < 2:
            return None

        last_pivot = pivots[-1]
        prev_pivot = pivots[-2]
        current_price = float(df['close'].iloc[-1])

        # 스윙 범위 계산
        swing_range = abs(last_pivot['price'] - prev_pivot['price'])

        # 기본 방향 예측 (마지막 피벗의 반대)
        if last_pivot['type'] == 'high':
            base_direction = 'down'
            base_target = last_pivot['price'] - swing_range * 0.618
        else:
            base_direction = 'up'
            base_target = last_pivot['price'] + swing_range * 0.618

        # 와이코프 구조에 따른 조정
        prob_adjustment = 0

        if wyckoff.phase == 'accumulation':
            if base_direction == 'up':
                prob_adjustment += 0.15
            if wyckoff.spring_detected:
                prob_adjustment += 0.2

        elif wyckoff.phase == 'distribution':
            if base_direction == 'down':
                prob_adjustment += 0.15
            if wyckoff.upthrust_detected:
                prob_adjustment += 0.2

        elif wyckoff.phase == 'markup':
            if base_direction == 'up':
                prob_adjustment += 0.1

        elif wyckoff.phase == 'markdown':
            if base_direction == 'down':
                prob_adjustment += 0.1

        # 다이버전스에 따른 조정
        for div in divergences:
            if div['type'].startswith('bullish') and base_direction == 'up':
                prob_adjustment += 0.15 * div['strength']
            elif div['type'].startswith('bearish') and base_direction == 'down':
                prob_adjustment += 0.15 * div['strength']

        # 피보나치 타겟 선택
        if base_direction == 'up':
            fib_base = last_pivot['price']  # 저점
            fib_range = prev_pivot['price'] - last_pivot['price']  # 이전 고점까지
            targets = [fib_base + fib_range * fib for fib in self.fib_levels]
        else:
            fib_base = last_pivot['price']  # 고점
            fib_range = last_pivot['price'] - prev_pivot['price']  # 이전 저점까지
            targets = [fib_base - fib_range * fib for fib in self.fib_levels]

        # 가장 가까운 피보나치 레벨 선택
        best_target = targets[1]  # 0.5 레벨 기본
        best_fib = self.fib_levels[1]

        for i, (target, fib) in enumerate(zip(targets, self.fib_levels)):
            distance = abs(target - current_price)
            if distance > swing_range * 0.3:  # 최소 거리 확보
                best_target = target
                best_fib = fib
                break

        # 예상 소요 시간 (평균 스윙 기간)
        if len(pivots) >= 3:
            avg_swing_bars = np.mean([
                pivots[i]['idx'] - pivots[i-1]['idx']
                for i in range(1, len(pivots))
            ])
        else:
            avg_swing_bars = 20

        # 최종 확률
        base_prob = 0.5
        final_prob = min(0.85, max(0.2, base_prob + prob_adjustment))

        # 신뢰도 = 와이코프 신뢰도 × 다이버전스 존재 여부
        div_bonus = 0.2 if divergences else 0
        confidence = wyckoff.confidence * 0.7 + div_bonus + 0.1

        return ZigzagPrediction(
            next_swing_direction=base_direction,
            target_price=best_target,
            probability=final_prob,
            expected_bars=int(avg_swing_bars),
            fib_level=best_fib,
            confidence=confidence
        )

    # =========================================================================
    # 5. 상위 타임프레임 추세 확인
    # =========================================================================

    def _check_higher_tf_trend(self, df: pd.DataFrame) -> str:
        """
        상위 타임프레임 추세 확인 (EMA 50/200 + 구조)
        """
        close = df['close']

        # EMA 계산
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        current = close.iloc[-1]
        ema20_val = ema20.iloc[-1]
        ema50_val = ema50.iloc[-1]
        ema200_val = ema200.iloc[-1]

        # 추세 점수
        score = 0

        if current > ema20_val:
            score += 1
        else:
            score -= 1

        if current > ema50_val:
            score += 1
        else:
            score -= 1

        if current > ema200_val:
            score += 1
        else:
            score -= 1

        if ema20_val > ema50_val:
            score += 0.5
        else:
            score -= 0.5

        if ema50_val > ema200_val:
            score += 0.5
        else:
            score -= 0.5

        if score >= 2:
            return 'bullish'
        elif score <= -2:
            return 'bearish'
        else:
            return 'neutral'

    # =========================================================================
    # 6. 종합 분석 및 신호 생성
    # =========================================================================

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """전체 분석 수행"""

        # 1. 와이코프 구조 분석
        wyckoff = self._analyze_wyckoff_structure(df)

        # 2. Stoch RSI 필터링된 다이버전스
        divergences = self._find_stoch_filtered_divergences(df)

        # 3. 지그재그 예측
        zigzag_pred = self._predict_next_zigzag(df, wyckoff, divergences)

        # 4. 상위 TF 추세
        htf_trend = self._check_higher_tf_trend(df)

        # 5. ATR
        atr = self._calculate_atr(df)
        current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0

        current_price = float(df['close'].iloc[-1])

        return {
            'current_price': current_price,
            'wyckoff': wyckoff,
            'divergences': divergences,
            'zigzag_prediction': zigzag_pred,
            'htf_trend': htf_trend,
            'atr': current_atr
        }

    def get_signal(self, df: pd.DataFrame) -> Optional[NavigatorSignal]:
        """매매 신호 생성"""

        analysis = self.analyze(df)

        wyckoff = analysis['wyckoff']
        divergences = analysis['divergences']
        zigzag = analysis['zigzag_prediction']
        htf_trend = analysis['htf_trend']
        atr = analysis['atr']
        current_price = analysis['current_price']

        if not zigzag:
            return None

        reasons = []

        # 롱 조건
        long_score = 0

        if zigzag.next_swing_direction == 'up':
            long_score += 2
            reasons.append(f"지그재그 상승 예측 ({zigzag.probability:.0%})")

        if wyckoff.phase == 'accumulation':
            long_score += 1.5
            reasons.append("와이코프 매집 구간")
        if wyckoff.spring_detected:
            long_score += 1
            reasons.append("스프링 감지")

        bullish_divs = [d for d in divergences if d['type'].startswith('bullish')]
        if bullish_divs:
            long_score += 1.5 * max(d['strength'] for d in bullish_divs)
            reasons.append(f"Bullish 다이버전스 (Stoch RSI 필터)")

        if htf_trend == 'bullish':
            long_score += 1
            reasons.append("상위 TF 상승 추세")
        elif htf_trend == 'bearish':
            long_score -= 1  # 역추세 페널티

        # 숏 조건
        short_score = 0

        if zigzag.next_swing_direction == 'down':
            short_score += 2

        if wyckoff.phase == 'distribution':
            short_score += 1.5
        if wyckoff.upthrust_detected:
            short_score += 1

        bearish_divs = [d for d in divergences if d['type'].startswith('bearish')]
        if bearish_divs:
            short_score += 1.5 * max(d['strength'] for d in bearish_divs)

        if htf_trend == 'bearish':
            short_score += 1
        elif htf_trend == 'bullish':
            short_score -= 1

        # 신호 결정
        min_score = 3.5  # 최소 점수

        # 추가 필터: 다이버전스가 있으면 보너스 (필수 아님)
        has_bullish_div = len([d for d in divergences if d['type'].startswith('bullish')]) > 0
        has_bearish_div = len([d for d in divergences if d['type'].startswith('bearish')]) > 0

        # 다이버전스 보너스
        if has_bullish_div:
            long_score += 0.5
        if has_bearish_div:
            short_score += 0.5

        if long_score >= min_score and long_score > short_score:
            # 롱 신호
            sl = current_price - atr * self.atr_sl_mult
            tp = zigzag.target_price

            # TP가 너무 가까우면 ATR 기반으로 조정
            if tp < current_price * 1.01:
                tp = current_price + atr * self.atr_tp_mult

            # 손익비 체크
            risk = current_price - sl
            reward = tp - current_price
            if risk > 0 and reward / risk < 1.5:
                return None  # 손익비 부족

            return NavigatorSignal(
                action='LONG',
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                position_size_pct=0.15,
                confidence=zigzag.confidence,
                reasons=reasons
            )

        elif short_score >= min_score and short_score > long_score:
            # 숏 신호
            sl = current_price + atr * self.atr_sl_mult
            tp = zigzag.target_price

            if tp > current_price * 0.99:
                tp = current_price - atr * self.atr_tp_mult

            risk = sl - current_price
            reward = current_price - tp
            if risk > 0 and reward / risk < 1.5:
                return None

            short_reasons = []
            if zigzag.next_swing_direction == 'down':
                short_reasons.append(f"지그재그 하락 예측 ({zigzag.probability:.0%})")
            if wyckoff.phase == 'distribution':
                short_reasons.append("와이코프 분산 구간")
            if wyckoff.upthrust_detected:
                short_reasons.append("업쓰러스트 감지")
            if bearish_divs:
                short_reasons.append("Bearish 다이버전스 (Stoch RSI 필터)")
            if htf_trend == 'bearish':
                short_reasons.append("상위 TF 하락 추세")

            return NavigatorSignal(
                action='SHORT',
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                position_size_pct=0.15,
                confidence=zigzag.confidence,
                reasons=short_reasons
            )

        return None


# 테스트
if __name__ == "__main__":
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(REPO_ROOT / 'src'))

    from wpcn.data.loaders import load_parquet

    data_path = REPO_ROOT / 'data' / 'raw' / 'binance_BTC-USDT_15m_90d.parquet'
    if data_path.exists():
        df = load_parquet(str(data_path))
        print(f"데이터: {len(df)} 캔들")

        nav = ChartNavigatorV2()
        analysis = nav.analyze(df)
        signal = nav.get_signal(df)

        print("\n" + "="*60)
        print("차트 네비게이터 V2 분석")
        print("="*60)

        print(f"\n현재가: ${analysis['current_price']:,.2f}")

        w = analysis['wyckoff']
        print(f"\n[와이코프 구조]")
        print(f"  페이즈: {w.phase.upper()}")
        print(f"  박스: ${w.box_low:,.0f} ~ ${w.box_high:,.0f}")
        print(f"  볼륨: {w.volume_profile}")
        print(f"  스프링: {'Yes' if w.spring_detected else 'No'}")
        print(f"  업쓰러스트: {'Yes' if w.upthrust_detected else 'No'}")

        print(f"\n[다이버전스] (Stoch RSI 필터)")
        if analysis['divergences']:
            for div in analysis['divergences']:
                print(f"  {div['type']}: RSI={div['rsi']:.1f}, Stoch={div['stoch']:.1f}, 강도={div['strength']:.2f}")
        else:
            print("  없음")

        z = analysis['zigzag_prediction']
        if z:
            print(f"\n[지그재그 예측]")
            print(f"  방향: {z.next_swing_direction.upper()}")
            print(f"  타겟: ${z.target_price:,.2f} (Fib {z.fib_level})")
            print(f"  확률: {z.probability:.1%}")
            print(f"  예상 봉수: {z.expected_bars}")

        print(f"\n[상위 TF 추세]: {analysis['htf_trend'].upper()}")
        print(f"[ATR]: ${analysis['atr']:,.2f}")

        if signal:
            print(f"\n{'='*60}")
            print(f"[신호] {signal.action}")
            print(f"{'='*60}")
            print(f"  진입: ${signal.entry_price:,.2f}")
            print(f"  손절: ${signal.stop_loss:,.2f}")
            print(f"  익절: ${signal.take_profit:,.2f}")
            print(f"  신뢰도: {signal.confidence:.1%}")
            print(f"\n  근거:")
            for r in signal.reasons:
                print(f"    - {r}")
        else:
            print("\n[신호] 없음 (조건 미충족)")
