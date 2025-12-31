"""
차트 네비게이터 V3 - 멀티타임프레임
Chart Navigator V3 - Multi-Timeframe

구조:
[4시간봉] 대추세 필터 (EMA 50/200)
    ↓
[1시간봉] 와이코프 구조 (매집/분산 박스)
    ↓
[15분봉] RSI 다이버전스 + 피보나치 레벨
    ↓
[5분봉] Stoch RSI 진입 타이밍

V3.1 개선사항:
- 15분봉/1시간봉 ATR 기반 TP/SL 계산 (5분봉 ATR 대신)
- DynamicParameterEngine 통합 (변동성 기반 동적 ATR 배수)
- 신호 필터 조건 완화 (min_score 4.0 → 3.5, min_alignment 2 → 1)
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.wpcn.features.dynamic_params import DynamicParameterEngine, DynamicParams


@dataclass
class MTFAnalysis:
    """멀티타임프레임 분석 결과"""
    # 4시간봉
    h4_trend: str           # 'bullish', 'bearish', 'neutral'
    h4_ema_position: str    # 'above', 'below', 'between'

    # 1시간봉
    # 'accumulation', 'distribution', 're_accumulation', 're_distribution',
    # 'markup', 'markdown', 'ranging'
    h1_wyckoff_phase: str
    h1_box_high: float
    h1_box_low: float
    h1_spring: bool
    h1_upthrust: bool

    # 15분봉
    m15_divergence: Optional[Dict]
    m15_fib_levels: List[float]
    m15_nearest_support: float
    m15_nearest_resistance: float

    # 5분봉
    m5_stoch_k: float
    m5_stoch_d: float
    m5_stoch_zone: str      # 'oversold', 'overbought', 'neutral'
    m5_reversal_candle: bool


@dataclass
class MTFSignal:
    """MTF 매매 신호"""
    action: str             # 'LONG', 'SHORT', 'WAIT'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timeframe_alignment: int  # 0~4 (몇 개 TF가 동의하는지)
    reasons: List[str]


class ChartNavigatorV3:
    """
    멀티타임프레임 차트 네비게이터

    5분봉 데이터를 받아서 상위 타임프레임으로 리샘플링하여 분석

    V3.1: DynamicParameterEngine 통합으로 코인별 최적화된 파라미터 사용
    """

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        stoch_k: int = 3,
        stoch_d: int = 3,
        wyckoff_lookback: int = 50,
        atr_period: int = 14,
        use_dynamic_params: bool = True,  # 동적 파라미터 사용 여부
    ):
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.stoch_k_smooth = stoch_k
        self.stoch_d_smooth = stoch_d
        self.wyckoff_lookback = wyckoff_lookback
        self.atr_period = atr_period
        self.use_dynamic_params = use_dynamic_params

        # 동적 파라미터 엔진
        self.dynamic_engine = DynamicParameterEngine(
            vol_lookback=100,
            fft_min_period=5,
            fft_max_period=100,
            hilbert_smooth=5
        )

        # 캐시
        self._cache = {}
        self._last_cache_time = None

    # =========================================================================
    # 리샘플링
    # =========================================================================

    def _resample_ohlcv(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """OHLCV 리샘플링"""
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return resampled

    def _prepare_mtf_data(self, df_5m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """5분봉 → 15분, 1시간, 4시간 리샘플링"""
        return {
            '5m': df_5m,
            '15m': self._resample_ohlcv(df_5m, '15min'),
            '1h': self._resample_ohlcv(df_5m, '1h'),
            '4h': self._resample_ohlcv(df_5m, '4h')
        }

    # =========================================================================
    # 지표 계산
    # =========================================================================

    def _calculate_ema(self, close: pd.Series, period: int) -> pd.Series:
        return close.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calculate_stoch_rsi(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        rsi = self._calculate_rsi(close)

        rsi_min = rsi.rolling(self.stoch_period).min()
        rsi_max = rsi.rolling(self.stoch_period).max()

        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100

        k = stoch_rsi.rolling(self.stoch_k_smooth).mean()
        d = k.rolling(self.stoch_d_smooth).mean()

        return k, d

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    # =========================================================================
    # 4시간봉: 대추세 분석
    # =========================================================================

    def _analyze_4h_trend(self, df_4h: pd.DataFrame) -> Dict[str, Any]:
        """4시간봉 대추세 분석"""
        if len(df_4h) < 200:
            return {'trend': 'neutral', 'ema_position': 'between', 'strength': 0}

        close = df_4h['close']
        current = close.iloc[-1]

        ema50 = self._calculate_ema(close, 50).iloc[-1]
        ema200 = self._calculate_ema(close, 200).iloc[-1]

        # EMA 위치
        if current > ema50 and current > ema200:
            ema_position = 'above'
        elif current < ema50 and current < ema200:
            ema_position = 'below'
        else:
            ema_position = 'between'

        # 추세 판정
        if ema50 > ema200 and current > ema50:
            trend = 'bullish'
            strength = min(1.0, (current - ema200) / ema200 * 10)
        elif ema50 < ema200 and current < ema50:
            trend = 'bearish'
            strength = min(1.0, (ema200 - current) / ema200 * 10)
        else:
            trend = 'neutral'
            strength = 0.3

        return {
            'trend': trend,
            'ema_position': ema_position,
            'ema50': ema50,
            'ema200': ema200,
            'strength': strength
        }

    # =========================================================================
    # 1시간봉: 와이코프 구조 분석
    # =========================================================================

    def _analyze_1h_wyckoff(self, df_1h: pd.DataFrame, h4_trend: str = 'neutral') -> Dict[str, Any]:
        """
        1시간봉 와이코프 구조 분석

        전통적 패턴:
        - accumulation: 바닥권 매집 (Spring)
        - distribution: 고점권 분산 (UTAD)

        추세 중 패턴:
        - re_accumulation: 상승 추세 중 횡보 + Higher Lows
        - re_distribution: 하락 추세 중 횡보 + Lower Highs
        """
        lookback = min(self.wyckoff_lookback, len(df_1h) - 1)
        if lookback < 20:
            return {
                'phase': 'ranging',
                'box_high': df_1h['high'].iloc[-1],
                'box_low': df_1h['low'].iloc[-1],
                'spring': False,
                'upthrust': False
            }

        recent = df_1h.iloc[-lookback:]

        box_high = recent['high'].max()
        box_low = recent['low'].min()
        box_mid = (box_high + box_low) / 2
        box_range = (box_high - box_low) / box_low

        current = df_1h['close'].iloc[-1]
        price_position = (current - box_low) / (box_high - box_low + 1e-10)

        # 스프링 감지
        spring = False
        for i in range(-10, 0):
            if i >= -len(recent):
                if recent['low'].iloc[i] < box_low * 0.995:
                    if recent['close'].iloc[i:].max() > box_mid:
                        spring = True
                        break

        # 업쓰러스트 감지
        upthrust = False
        for i in range(-10, 0):
            if i >= -len(recent):
                if recent['high'].iloc[i] > box_high * 1.005:
                    if recent['close'].iloc[i:].min() < box_mid:
                        upthrust = True
                        break

        # === Re-Accumulation / Re-Distribution 감지 ===
        # Higher Lows / Lower Highs 체크 (최근 lookback의 절반씩 비교)
        half = lookback // 2
        if half >= 10:
            first_half = recent.iloc[:half]
            second_half = recent.iloc[half:]

            first_half_low = first_half['low'].min()
            second_half_low = second_half['low'].min()
            has_higher_lows = second_half_low > first_half_low * 1.005  # 0.5% 이상 높아야

            first_half_high = first_half['high'].max()
            second_half_high = second_half['high'].max()
            has_lower_highs = second_half_high < first_half_high * 0.995  # 0.5% 이상 낮아야
        else:
            has_higher_lows = False
            has_lower_highs = False

        # 거래량 패턴 (조용한 흡수형 = 평균 이하)
        vol_ma = df_1h['volume'].rolling(20).mean()
        recent_vol = df_1h['volume'].iloc[-10:].mean()
        quiet_volume = recent_vol < vol_ma.iloc[-1] * 1.2 if not pd.isna(vol_ma.iloc[-1]) else True

        # 페이즈 판정
        if box_range < 0.05:  # 5% 이내 횡보
            # === Re-Accumulation: 상승 추세 중 + Higher Lows + 조용한 거래량 ===
            if h4_trend == 'bullish' and has_higher_lows and quiet_volume:
                phase = 're_accumulation'
            # === Re-Distribution: 하락 추세 중 + Lower Highs + 조용한 거래량 ===
            elif h4_trend == 'bearish' and has_lower_highs and quiet_volume:
                phase = 're_distribution'
            # === 전통적 패턴 ===
            elif spring:
                phase = 'accumulation'
            elif upthrust:
                phase = 'distribution'
            else:
                phase = 'ranging'
        else:
            if price_position > 0.7:
                phase = 'markup'
            elif price_position < 0.3:
                phase = 'markdown'
            else:
                phase = 'ranging'

        return {
            'phase': phase,
            'box_high': box_high,
            'box_low': box_low,
            'box_mid': box_mid,
            'spring': spring,
            'upthrust': upthrust,
            'price_position': price_position,
            'has_higher_lows': has_higher_lows,
            'has_lower_highs': has_lower_highs
        }

    # =========================================================================
    # 15분봉: RSI 다이버전스 + 피보나치
    # =========================================================================

    def _analyze_15m_divergence(self, df_15m: pd.DataFrame) -> Dict[str, Any]:
        """15분봉 다이버전스 분석 (Stoch RSI 필터)"""
        close = df_15m['close']
        rsi = self._calculate_rsi(close)
        stoch_k, stoch_d = self._calculate_stoch_rsi(close)

        divergence = None

        # 피벗 찾기
        window = 5
        pivots_low = []
        pivots_high = []

        for i in range(window, len(df_15m) - window):
            stoch_val = stoch_k.iloc[i]

            # 과매도 구간에서만 저점 체크
            if stoch_val < 30:
                if close.iloc[i] == close.iloc[i-window:i+window+1].min():
                    pivots_low.append({
                        'idx': i,
                        'price': close.iloc[i],
                        'rsi': rsi.iloc[i],
                        'stoch': stoch_val
                    })

            # 과매수 구간에서만 고점 체크
            if stoch_val > 70:
                if close.iloc[i] == close.iloc[i-window:i+window+1].max():
                    pivots_high.append({
                        'idx': i,
                        'price': close.iloc[i],
                        'rsi': rsi.iloc[i],
                        'stoch': stoch_val
                    })

        # Bullish Divergence
        recent_lows = [p for p in pivots_low if p['idx'] > len(df_15m) - 50]
        if len(recent_lows) >= 2:
            prev, curr = recent_lows[-2], recent_lows[-1]
            if curr['price'] < prev['price'] and curr['rsi'] > prev['rsi']:
                divergence = {
                    'type': 'bullish',
                    'strength': min(1.0, (curr['rsi'] - prev['rsi']) / 20),
                    'price': curr['price'],
                    'idx': curr['idx']
                }

        # Bearish Divergence
        recent_highs = [p for p in pivots_high if p['idx'] > len(df_15m) - 50]
        if len(recent_highs) >= 2:
            prev, curr = recent_highs[-2], recent_highs[-1]
            if curr['price'] > prev['price'] and curr['rsi'] < prev['rsi']:
                div = {
                    'type': 'bearish',
                    'strength': min(1.0, (prev['rsi'] - curr['rsi']) / 20),
                    'price': curr['price'],
                    'idx': curr['idx']
                }
                # 기존 다이버전스보다 강하면 교체
                if divergence is None or div['strength'] > divergence['strength']:
                    divergence = div

        # 피보나치 레벨
        lookback = min(100, len(df_15m))
        recent = df_15m.iloc[-lookback:]
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()

        fib_levels = {
            0.236: swing_low + (swing_high - swing_low) * 0.236,
            0.382: swing_low + (swing_high - swing_low) * 0.382,
            0.5: swing_low + (swing_high - swing_low) * 0.5,
            0.618: swing_low + (swing_high - swing_low) * 0.618,
            0.786: swing_low + (swing_high - swing_low) * 0.786
        }

        current = close.iloc[-1]

        # 가장 가까운 지지/저항
        supports = [v for v in fib_levels.values() if v < current]
        resistances = [v for v in fib_levels.values() if v > current]

        nearest_support = max(supports) if supports else swing_low
        nearest_resistance = min(resistances) if resistances else swing_high

        return {
            'divergence': divergence,
            'fib_levels': fib_levels,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'swing_high': swing_high,
            'swing_low': swing_low
        }

    # =========================================================================
    # 5분봉: Stoch RSI 진입 타이밍
    # =========================================================================

    def _analyze_5m_entry(self, df_5m: pd.DataFrame) -> Dict[str, Any]:
        """5분봉 진입 타이밍 분석"""
        close = df_5m['close']
        stoch_k, stoch_d = self._calculate_stoch_rsi(close)

        k_val = stoch_k.iloc[-1]
        d_val = stoch_d.iloc[-1]
        k_prev = stoch_k.iloc[-2] if len(stoch_k) > 1 else k_val

        # 존 판정
        if k_val < 20:
            zone = 'oversold'
        elif k_val > 80:
            zone = 'overbought'
        else:
            zone = 'neutral'

        # 반전 캔들 감지
        reversal_candle = False
        if len(df_5m) >= 3:
            c1 = df_5m.iloc[-3]
            c2 = df_5m.iloc[-2]
            c3 = df_5m.iloc[-1]

            # Bullish reversal (망치형, 장악형)
            if zone == 'oversold':
                # 망치형
                body = abs(c3['close'] - c3['open'])
                lower_wick = min(c3['open'], c3['close']) - c3['low']
                if lower_wick > body * 2 and c3['close'] > c3['open']:
                    reversal_candle = True
                # 상승 장악형
                if c2['close'] < c2['open'] and c3['close'] > c3['open']:
                    if c3['close'] > c2['open'] and c3['open'] < c2['close']:
                        reversal_candle = True

            # Bearish reversal (역망치형, 하락장악형)
            if zone == 'overbought':
                body = abs(c3['close'] - c3['open'])
                upper_wick = c3['high'] - max(c3['open'], c3['close'])
                if upper_wick > body * 2 and c3['close'] < c3['open']:
                    reversal_candle = True
                if c2['close'] > c2['open'] and c3['close'] < c3['open']:
                    if c3['close'] < c2['open'] and c3['open'] > c2['close']:
                        reversal_candle = True

        # K가 D를 크로스
        k_cross_up = k_prev < d_val and k_val > d_val
        k_cross_down = k_prev > d_val and k_val < d_val

        return {
            'stoch_k': k_val,
            'stoch_d': d_val,
            'zone': zone,
            'reversal_candle': reversal_candle,
            'k_cross_up': k_cross_up,
            'k_cross_down': k_cross_down
        }

    # =========================================================================
    # 종합 분석 및 신호
    # =========================================================================

    def analyze(self, df_5m: pd.DataFrame) -> Dict[str, Any]:
        """전체 MTF 분석"""
        # 리샘플링
        mtf_data = self._prepare_mtf_data(df_5m)

        # 각 타임프레임 분석 (4시간봉 추세를 1시간봉에 전달)
        h4_analysis = self._analyze_4h_trend(mtf_data['4h'])
        h1_analysis = self._analyze_1h_wyckoff(mtf_data['1h'], h4_analysis['trend'])
        m15_analysis = self._analyze_15m_divergence(mtf_data['15m'])
        m5_analysis = self._analyze_5m_entry(mtf_data['5m'])

        # ATR - 다중 타임프레임 (5분, 15분, 1시간)
        atr_5m = self._calculate_atr(mtf_data['5m'])
        atr_15m = self._calculate_atr(mtf_data['15m'])
        atr_1h = self._calculate_atr(mtf_data['1h'])

        current_atr_5m = float(atr_5m.iloc[-1]) if len(atr_5m) > 0 and not pd.isna(atr_5m.iloc[-1]) else 0
        current_atr_15m = float(atr_15m.iloc[-1]) if len(atr_15m) > 0 and not pd.isna(atr_15m.iloc[-1]) else 0
        current_atr_1h = float(atr_1h.iloc[-1]) if len(atr_1h) > 0 and not pd.isna(atr_1h.iloc[-1]) else 0

        current_price = float(df_5m['close'].iloc[-1])

        # 동적 파라미터 계산 (15분봉 기준)
        dynamic_params = None
        if self.use_dynamic_params and len(mtf_data['15m']) >= 50:
            try:
                dynamic_params = self.dynamic_engine.compute_dynamic_params(mtf_data['15m'])
            except Exception:
                dynamic_params = None

        return {
            'current_price': current_price,
            'atr': current_atr_5m,           # 하위 호환성 유지
            'atr_5m': current_atr_5m,
            'atr_15m': current_atr_15m,
            'atr_1h': current_atr_1h,
            'dynamic_params': dynamic_params,
            'h4': h4_analysis,
            'h1': h1_analysis,
            'm15': m15_analysis,
            'm5': m5_analysis
        }

    def get_signal(self, df_5m: pd.DataFrame) -> Optional[MTFSignal]:
        """MTF 매매 신호 생성"""
        analysis = self.analyze(df_5m)

        h4 = analysis['h4']
        h1 = analysis['h1']
        m15 = analysis['m15']
        m5 = analysis['m5']
        current_price = analysis['current_price']

        # === ATR 선택: 15분봉 ATR 우선 (더 안정적) ===
        atr_15m = analysis.get('atr_15m', 0)
        atr_5m = analysis.get('atr', 0)
        atr = atr_15m if atr_15m > 0 else atr_5m

        # === 동적 파라미터 적용 ===
        dynamic_params: Optional[DynamicParams] = analysis.get('dynamic_params')

        if dynamic_params and self.use_dynamic_params:
            # 동적 ATR 배수 사용
            sl_mult = dynamic_params.atr_mult_stop
            tp_mult = dynamic_params.atr_mult_tp1
        else:
            # 기본값 (15분봉 ATR 기준으로 조정됨)
            sl_mult = 1.5   # 15분봉 ATR × 1.5 ≈ 5분봉 ATR × 4.5
            tp_mult = 3.0   # 15분봉 ATR × 3.0 ≈ 5분봉 ATR × 9.0

        reasons = []
        alignment = 0

        # === 롱 조건 ===
        long_score = 0

        # 4시간봉: 상승 추세
        if h4['trend'] == 'bullish':
            long_score += 2
            alignment += 1
            reasons.append(f"4H 상승추세 (EMA {h4['ema_position']})")
        elif h4['trend'] == 'bearish':
            long_score -= 1.5  # 역추세 페널티 완화 (2.0 → 1.5)

        # 1시간봉: 매집 구간 (전통적 + 추세 중)
        if h1['phase'] == 'accumulation':
            long_score += 2
            alignment += 1
            reasons.append("1H 전통적 매집 구간")
        if h1['phase'] == 're_accumulation':
            long_score += 2.5  # 추세 순응이라 더 높은 점수
            alignment += 1
            reasons.append("1H 상승 중 매집 (Re-Accumulation)")
        if h1['spring']:
            long_score += 1.5
            reasons.append("1H 스프링 감지")
        if h1['phase'] == 'markup':
            long_score += 1.5  # 마크업 점수 상향 (1.0 → 1.5)
            alignment += 1
            reasons.append("1H 마크업 구간")

        # 15분봉: Bullish 다이버전스
        if m15['divergence'] and m15['divergence']['type'] == 'bullish':
            long_score += 2.5 * m15['divergence']['strength']  # 다이버전스 가중치 상향
            alignment += 1
            reasons.append(f"15M Bullish 다이버전스 (강도: {m15['divergence']['strength']:.2f})")

        # 5분봉: 과매도 + 반전
        if m5['zone'] == 'oversold':
            long_score += 1
            if m5['reversal_candle']:
                long_score += 1.5
                alignment += 1
                reasons.append("5M 과매도 + 반전캔들")
            if m5['k_cross_up']:
                long_score += 1
                reasons.append("5M Stoch K 상향돌파")

        # === 숏 조건 ===
        short_score = 0
        short_reasons = []
        short_alignment = 0

        # 4시간봉: 하락 추세
        if h4['trend'] == 'bearish':
            short_score += 2
            short_alignment += 1
            short_reasons.append(f"4H 하락추세 (EMA {h4['ema_position']})")
        elif h4['trend'] == 'bullish':
            short_score -= 1.5  # 역추세 페널티 완화

        # 1시간봉: 분산 구간 (전통적 + 추세 중)
        if h1['phase'] == 'distribution':
            short_score += 2
            short_alignment += 1
            short_reasons.append("1H 전통적 분산 구간")
        if h1['phase'] == 're_distribution':
            short_score += 2.5  # 추세 순응이라 더 높은 점수
            short_alignment += 1
            short_reasons.append("1H 하락 중 분산 (Re-Distribution)")
        if h1['upthrust']:
            short_score += 1.5
            short_reasons.append("1H 업쓰러스트 감지")
        if h1['phase'] == 'markdown':
            short_score += 1.5  # 마크다운 점수 상향
            short_alignment += 1
            short_reasons.append("1H 마크다운 구간")

        # 15분봉: Bearish 다이버전스
        if m15['divergence'] and m15['divergence']['type'] == 'bearish':
            short_score += 2.5 * m15['divergence']['strength']  # 다이버전스 가중치 상향
            short_alignment += 1
            short_reasons.append(f"15M Bearish 다이버전스 (강도: {m15['divergence']['strength']:.2f})")

        # 5분봉: 과매수 + 반전
        if m5['zone'] == 'overbought':
            short_score += 1
            if m5['reversal_candle']:
                short_score += 1.5
                short_alignment += 1
                short_reasons.append("5M 과매수 + 반전캔들")
            if m5['k_cross_down']:
                short_score += 1
                short_reasons.append("5M Stoch K 하향돌파")

        # === 신호 결정 (조건 완화) ===
        min_score = 3.5      # 4.0 → 3.5 (완화)
        min_alignment = 1    # 2 → 1 (완화)

        if long_score >= min_score and alignment >= min_alignment and long_score > short_score:
            # 롱 신호 - 15분봉 ATR 기반 TP/SL
            sl = current_price - atr * sl_mult
            tp = m15['nearest_resistance']

            # TP가 너무 가까우면 ATR 기반
            if tp < current_price * 1.01:
                tp = current_price + atr * tp_mult

            # 손익비 체크 (완화: 1.5 → 1.2)
            risk = current_price - sl
            reward = tp - current_price
            if risk > 0 and reward / risk < 1.2:
                return None

            return MTFSignal(
                action='LONG',
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                confidence=min(0.9, long_score / 10),
                timeframe_alignment=alignment,
                reasons=reasons
            )

        elif short_score >= min_score and short_alignment >= min_alignment and short_score > long_score:
            # 숏 신호 - 15분봉 ATR 기반 TP/SL
            sl = current_price + atr * sl_mult
            tp = m15['nearest_support']

            if tp > current_price * 0.99:
                tp = current_price - atr * tp_mult

            risk = sl - current_price
            reward = current_price - tp
            if risk > 0 and reward / risk < 1.2:
                return None

            return MTFSignal(
                action='SHORT',
                entry_price=current_price,
                stop_loss=sl,
                take_profit=tp,
                confidence=min(0.9, short_score / 10),
                timeframe_alignment=short_alignment,
                reasons=short_reasons
            )

        return None


# 테스트
if __name__ == "__main__":
    print("ChartNavigatorV3 - MTF 분석기 로드 완료")
    print("5분봉 데이터를 입력하면 4H/1H/15M/5M 분석을 수행합니다.")
