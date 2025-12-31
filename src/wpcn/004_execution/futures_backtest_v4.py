"""
선물 백테스트 엔진 V4 - 완전한 MTF 전략 구현
Futures Backtest Engine V4 - Full MTF Strategy Implementation

STRATEGY_ARCHITECTURE.md 기반 완전 구현:
- 4시간봉: 대추세 필터 (EMA 50/200)
- 1시간봉: Wyckoff 구조 (Accumulation/Distribution/Re-Accumulation/Re-Distribution)
- 15분봉: RSI 다이버전스 + 피보나치
- 5분봉: 진입 타이밍 (Stoch RSI + 반전 캔들)

점수 시스템:
- min_score >= 4.0
- TF alignment >= 2
- RR ratio >= 1.5
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd

# 기존 모듈
from wpcn.features.indicators import atr, rsi, stoch_rsi, detect_rsi_divergence, detect_fib_bounce


@dataclass
class MTFConfig:
    """MTF 백테스트 설정"""
    # 레버리지
    leverage: float = 5.0

    # 포지션
    position_pct: float = 0.15  # 15%

    # 손익비
    min_rr_ratio: float = 1.5

    # ATR 기반 SL/TP
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 4.0

    # 신호 조건
    min_score: float = 4.0
    min_tf_alignment: int = 2

    # 시간 설정 (5분봉 기준)
    warmup_bars: int = 1000  # 워밍업
    analysis_interval: int = 24  # 분석 간격 (2시간)
    cooldown_bars: int = 72  # 쿨다운 (6시간)
    max_hold_bars: int = 144  # 최대 보유 (12시간)

    # EMA 설정 (4시간봉)
    ema_short: int = 50
    ema_long: int = 200

    # Wyckoff 설정 (1시간봉)
    box_lookback: int = 50
    spring_threshold: float = 0.005  # 0.5%

    # RSI 다이버전스 설정 (15분봉)
    rsi_period: int = 14
    stoch_rsi_period: int = 14
    div_lookback: int = 20

    # 비용
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage: float = 0.0001


@dataclass
class Position:
    """포지션"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    margin: float
    leverage: float
    tp_price: float
    sl_price: float
    entry_bar: int
    entry_time: datetime
    score: float
    tf_alignment: int
    reasons: Dict[str, str]


class MTFBacktestEngine:
    """
    MTF 백테스트 엔진 V4

    STRATEGY_ARCHITECTURE.md 완전 구현
    """

    def __init__(self, config: MTFConfig = None):
        self.config = config or MTFConfig()
        self.position: Optional[Position] = None
        self.trades: List[Dict] = []
        self.signals: List[Dict] = []
        self.equity_history: List[Dict] = []

    def run(
        self,
        df_5m: pd.DataFrame,
        symbol: str = "BTC/USDT",
        initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        """백테스트 실행"""
        cfg = self.config

        # 초기화
        self.position = None
        self.trades = []
        self.signals = []
        self.equity_history = []

        cash = initial_capital
        last_entry_bar = -cfg.cooldown_bars - 1
        cached_signal = None

        # 리샘플링
        df_15m = self._resample(df_5m, '15min')
        df_1h = self._resample(df_5m, '1h')
        df_4h = self._resample(df_5m, '4h')

        # 지표 사전 계산 (4시간봉)
        df_4h['ema_short'] = df_4h['close'].ewm(span=cfg.ema_short).mean()
        df_4h['ema_long'] = df_4h['close'].ewm(span=cfg.ema_long).mean()

        # ATR 계산
        atr_5m = atr(df_5m, 14)
        atr_15m = atr(df_15m, 14)

        # RSI 다이버전스 (15분봉)
        div_15m = detect_rsi_divergence(df_15m, rsi_len=cfg.rsi_period, lookback=cfg.div_lookback)

        # 피보나치 반등 (15분봉)
        fib_15m = detect_fib_bounce(df_15m, swing_lookback=100, tolerance=0.005)

        # Stoch RSI (5분봉)
        stoch_k, stoch_d = stoch_rsi(df_5m['close'], cfg.stoch_rsi_period, cfg.stoch_rsi_period, 3, 3)

        n = len(df_5m)
        idx = df_5m.index

        print(f"Running MTF backtest: {n} bars, {symbol}")

        for i in range(cfg.warmup_bars, n):
            t = idx[i]
            row = df_5m.iloc[i]
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            current_atr = atr_5m.iloc[i] if i < len(atr_5m) and not pd.isna(atr_5m.iloc[i]) else c * 0.01

            # === 1. 포지션 청산 체크 ===
            if self.position:
                exit_result = self._check_exit(self.position, h, l, c, i, t, current_atr)
                if exit_result:
                    exit_price, exit_reason, pnl = exit_result
                    cash += self.position.margin + pnl

                    self.trades.append({
                        'symbol': symbol,
                        'side': self.position.side,
                        'entry_price': self.position.entry_price,
                        'exit_price': exit_price,
                        'quantity': self.position.quantity,
                        'tp_price': self.position.tp_price,
                        'sl_price': self.position.sl_price,
                        'pnl': pnl,
                        'pnl_pct': pnl / self.position.margin * 100,
                        'entry_time': self.position.entry_time,
                        'exit_time': t,
                        'exit_reason': exit_reason,
                        'score': self.position.score,
                        'tf_alignment': self.position.tf_alignment,
                        'reasons': self.position.reasons,
                        'hold_bars': i - self.position.entry_bar
                    })
                    self.position = None

            # === 2. 분석 주기 체크 (2시간마다) ===
            if i % cfg.analysis_interval == 0:
                signal = self._analyze_mtf(
                    df_5m, df_15m, df_1h, df_4h,
                    div_15m, fib_15m, stoch_k, stoch_d,
                    atr_15m, i, t, symbol
                )
                if signal and signal['action'] != 'none':
                    cached_signal = signal
                    self.signals.append(signal)

            # === 3. 진입 조건 체크 ===
            if (self.position is None and
                cached_signal is not None and
                i - last_entry_bar >= cfg.cooldown_bars and
                cash > 0):

                # 포지션 오픈
                entry_price = c * (1 + cfg.slippage if cached_signal['action'] == 'long' else 1 - cfg.slippage)

                # TP/SL 계산 (항상 ATR 기반)
                if cached_signal['action'] == 'long':
                    sl_price = entry_price - current_atr * cfg.sl_atr_mult
                    tp_price = entry_price + current_atr * cfg.tp_atr_mult
                else:
                    sl_price = entry_price + current_atr * cfg.sl_atr_mult
                    tp_price = entry_price - current_atr * cfg.tp_atr_mult

                # RR 체크
                risk = abs(entry_price - sl_price)
                reward = abs(tp_price - entry_price)
                rr = reward / risk if risk > 0 else 0

                if rr >= cfg.min_rr_ratio:
                    # 포지션 크기 계산
                    position_value = cash * cfg.position_pct * cfg.leverage
                    quantity = position_value / entry_price
                    margin = position_value / cfg.leverage
                    fee = position_value * cfg.maker_fee

                    self.position = Position(
                        symbol=symbol,
                        side=cached_signal['action'],
                        entry_price=entry_price,
                        quantity=quantity,
                        margin=margin,
                        leverage=cfg.leverage,
                        tp_price=tp_price,
                        sl_price=sl_price,
                        entry_bar=i,
                        entry_time=t,
                        score=cached_signal['score'],
                        tf_alignment=cached_signal['tf_alignment'],
                        reasons=cached_signal['reasons']
                    )

                    cash -= margin + fee
                    last_entry_bar = i
                    cached_signal = None

            # === Equity 기록 ===
            equity = self._calc_equity(cash, c)
            self.equity_history.append({
                'time': t,
                'equity': equity,
                'cash': cash,
                'position': 1 if self.position else 0
            })

        # 결과 생성
        return self._generate_result(symbol, initial_capital)

    def _resample(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """리샘플링"""
        return df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    def _analyze_mtf(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        div_15m: pd.DataFrame,
        fib_15m: pd.DataFrame,
        stoch_k: pd.Series,
        stoch_d: pd.Series,
        atr_15m: pd.Series,
        bar_idx: int,
        timestamp: datetime,
        symbol: str
    ) -> Optional[Dict]:
        """MTF 분석 (완료된 봉만 사용 - look-ahead bias 방지)"""
        from datetime import timedelta

        cfg = self.config

        long_score = 0.0
        short_score = 0.0
        reasons = {}
        tf_alignment = 0

        # 중요: 완료된 이전 봉만 참조해야 look-ahead bias 방지
        # 현재 시간의 봉은 아직 진행 중이므로, 이전 완료된 봉을 사용
        t_15m = timestamp.floor('15min') - timedelta(minutes=15)
        t_1h = timestamp.floor('1h') - timedelta(hours=1)
        t_4h = timestamp.floor('4h') - timedelta(hours=4)

        # === 4시간봉: 대추세 필터 ===
        if t_4h in df_4h.index:
            row_4h = df_4h.loc[t_4h]
            close_4h = row_4h['close']
            ema_short = row_4h.get('ema_short', close_4h)
            ema_long = row_4h.get('ema_long', close_4h)

            if not pd.isna(ema_short) and not pd.isna(ema_long):
                if close_4h > ema_short > ema_long:
                    long_score += 2.0
                    short_score -= 2.0  # 역추세 페널티
                    reasons['4h_trend'] = 'BULLISH'
                    tf_alignment += 1
                elif close_4h < ema_short < ema_long:
                    short_score += 2.0
                    long_score -= 2.0  # 역추세 페널티
                    reasons['4h_trend'] = 'BEARISH'
                    tf_alignment += 1
                else:
                    reasons['4h_trend'] = 'NEUTRAL'

        # === 1시간봉: Wyckoff 구조 ===
        if t_1h in df_1h.index:
            wyckoff_result = self._analyze_wyckoff(df_1h, t_1h)

            if wyckoff_result['phase'] == 'accumulation':
                long_score += 2.0
                reasons['1h_wyckoff'] = 'ACCUMULATION'
                tf_alignment += 1
            elif wyckoff_result['phase'] == 're_accumulation':
                long_score += 2.5
                reasons['1h_wyckoff'] = 'RE_ACCUMULATION'
                tf_alignment += 1
            elif wyckoff_result['phase'] == 'distribution':
                short_score += 2.0
                reasons['1h_wyckoff'] = 'DISTRIBUTION'
                tf_alignment += 1
            elif wyckoff_result['phase'] == 're_distribution':
                short_score += 2.5
                reasons['1h_wyckoff'] = 'RE_DISTRIBUTION'
                tf_alignment += 1
            elif wyckoff_result['phase'] == 'markup':
                long_score += 1.0
                reasons['1h_wyckoff'] = 'MARKUP'
            elif wyckoff_result['phase'] == 'markdown':
                short_score += 1.0
                reasons['1h_wyckoff'] = 'MARKDOWN'

            # 스프링/업쓰러스트
            if wyckoff_result.get('spring'):
                long_score += 1.5
                reasons['1h_event'] = 'SPRING'
            elif wyckoff_result.get('upthrust'):
                short_score += 1.5
                reasons['1h_event'] = 'UPTHRUST'

        # === 15분봉: RSI 다이버전스 + 피보나치 ===
        if t_15m in div_15m.index:
            div_row = div_15m.loc[t_15m]

            if div_row.get('bullish_div', False):
                strength = div_row.get('div_strength', 0.7)
                long_score += 2.0 * min(strength, 1.0)
                reasons['15m_divergence'] = f'BULLISH (str={strength:.2f})'
                tf_alignment += 1
            elif div_row.get('bearish_div', False):
                strength = div_row.get('div_strength', 0.7)
                short_score += 2.0 * min(strength, 1.0)
                reasons['15m_divergence'] = f'BEARISH (str={strength:.2f})'
                tf_alignment += 1

        # 피보나치 반등 (점수만 부여, TP는 ATR 기반으로만 사용)
        if t_15m in fib_15m.index:
            fib_row = fib_15m.loc[t_15m]

            if fib_row.get('fib_bounce_long', False):
                long_score += 1.0
                reasons['15m_fib'] = 'BOUNCE_LONG'
            elif fib_row.get('fib_bounce_short', False):
                short_score += 1.0
                reasons['15m_fib'] = 'BOUNCE_SHORT'

        # === 5분봉: 진입 타이밍 ===
        if bar_idx < len(stoch_k) and bar_idx < len(stoch_d):
            k_val = stoch_k.iloc[bar_idx]
            d_val = stoch_d.iloc[bar_idx]

            if not pd.isna(k_val) and not pd.isna(d_val):
                # 과매수/과매도
                if k_val < 20:
                    long_score += 1.0
                    reasons['5m_stoch'] = f'OVERSOLD (K={k_val:.1f})'
                    tf_alignment += 1
                elif k_val > 80:
                    short_score += 1.0
                    reasons['5m_stoch'] = f'OVERBOUGHT (K={k_val:.1f})'
                    tf_alignment += 1

                # K/D 크로스
                if bar_idx > 0:
                    k_prev = stoch_k.iloc[bar_idx - 1]
                    d_prev = stoch_d.iloc[bar_idx - 1]

                    if not pd.isna(k_prev) and not pd.isna(d_prev):
                        if k_prev < d_prev and k_val > d_val:  # 상향 돌파
                            long_score += 1.0
                            reasons['5m_cross'] = 'K_UP_CROSS'
                        elif k_prev > d_prev and k_val < d_val:  # 하향 돌파
                            short_score += 1.0
                            reasons['5m_cross'] = 'K_DOWN_CROSS'

        # 반전 캔들 체크 (5분봉)
        candle_result = self._check_reversal_candle(df_5m, bar_idx)
        if candle_result == 'bullish':
            long_score += 1.5
            reasons['5m_candle'] = 'HAMMER'
        elif candle_result == 'bearish':
            short_score += 1.5
            reasons['5m_candle'] = 'SHOOTING_STAR'

        # === 신호 결정 (방향 일관성 체크 강화) ===
        action = 'none'
        score = 0.0

        # Long 신호: 4H Bullish + (1H Accumulation or Re_Accumulation or Markup)
        # Short 신호: 4H Bearish + (1H Distribution or Re_Distribution or Markdown)
        long_consistent = (
            reasons.get('4h_trend') == 'BULLISH' and
            reasons.get('1h_wyckoff') in ['ACCUMULATION', 'RE_ACCUMULATION', 'MARKUP']
        )
        short_consistent = (
            reasons.get('4h_trend') == 'BEARISH' and
            reasons.get('1h_wyckoff') in ['DISTRIBUTION', 'RE_DISTRIBUTION', 'MARKDOWN']
        )

        if long_score >= cfg.min_score and long_score > short_score and tf_alignment >= cfg.min_tf_alignment and long_consistent:
            action = 'long'
            score = long_score
        elif short_score >= cfg.min_score and short_score > long_score and tf_alignment >= cfg.min_tf_alignment and short_consistent:
            action = 'short'
            score = short_score

        return {
            'symbol': symbol,
            'timestamp': timestamp,
            'action': action,
            'score': score,
            'long_score': long_score,
            'short_score': short_score,
            'tf_alignment': tf_alignment,
            'reasons': reasons
        }

    def _analyze_wyckoff(self, df_1h: pd.DataFrame, timestamp) -> Dict:
        """Wyckoff 구조 분석 (볼륨 포함)"""
        cfg = self.config

        # 최근 50봉 데이터
        try:
            idx = df_1h.index.get_loc(timestamp)
        except KeyError:
            return {'phase': 'unknown'}

        if idx < cfg.box_lookback:
            return {'phase': 'unknown'}

        window = df_1h.iloc[idx - cfg.box_lookback:idx + 1]

        box_high = window['high'].max()
        box_low = window['low'].min()
        box_range = box_high - box_low
        current_close = window['close'].iloc[-1]

        # Higher Lows / Lower Highs 체크
        recent_lows = window['low'].rolling(5).min().dropna()
        recent_highs = window['high'].rolling(5).max().dropna()

        higher_lows = len(recent_lows) >= 3 and recent_lows.iloc[-1] > recent_lows.iloc[-3]
        lower_highs = len(recent_highs) >= 3 and recent_highs.iloc[-1] < recent_highs.iloc[-3]

        # 가격 위치
        price_position = (current_close - box_low) / box_range if box_range > 0 else 0.5

        # === 볼륨 분석 (Wyckoff 핵심) ===
        if 'volume' in window.columns:
            avg_volume = window['volume'].mean()
            recent_volume = window['volume'].iloc[-5:].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

            # 하락 시 볼륨 감소 = 축적 신호
            down_bars = window[window['close'] < window['open']]
            up_bars = window[window['close'] >= window['open']]

            down_volume = down_bars['volume'].mean() if len(down_bars) > 0 else 0
            up_volume = up_bars['volume'].mean() if len(up_bars) > 0 else 0

            # 축적: 하락 볼륨 <= 상승 볼륨 (매도 압력이 매수보다 약함)
            accumulation_volume = down_volume <= up_volume if up_volume > 0 else False
            # 분배: 상승 볼륨 <= 하락 볼륨 (매수 압력이 매도보다 약함)
            distribution_volume = up_volume <= down_volume if down_volume > 0 else False
        else:
            accumulation_volume = False
            distribution_volume = False
            volume_ratio = 1.0

        # 스프링: 박스 하단 돌파 후 복귀 + 볼륨 급증
        spring = False
        recent_low = window['low'].iloc[-3:].min()
        if recent_low < box_low * (1 - cfg.spring_threshold) and current_close > box_low:
            # 스프링은 볼륨 급증과 함께 발생
            if 'volume' in window.columns:
                spring_volume = window['volume'].iloc[-3:].max()
                if spring_volume > avg_volume * 1.5:
                    spring = True
            else:
                spring = True

        # 업쓰러스트: 박스 상단 돌파 후 복귀 + 볼륨 급증
        upthrust = False
        recent_high = window['high'].iloc[-3:].max()
        if recent_high > box_high * (1 + cfg.spring_threshold) and current_close < box_high:
            if 'volume' in window.columns:
                upthrust_volume = window['volume'].iloc[-3:].max()
                if upthrust_volume > avg_volume * 1.5:
                    upthrust = True
            else:
                upthrust = True

        # 추세 판단 (EMA 기반)
        ema_20 = window['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = window['close'].ewm(span=50).mean().iloc[-1] if len(window) >= 50 else ema_20

        uptrend = current_close > ema_20 > ema_50
        downtrend = current_close < ema_20 < ema_50

        # 페이즈 결정 (더 엄격한 조건)
        phase = 'ranging'

        # 축적: 박스 하단 + Higher Lows + 볼륨 조건
        if spring:
            phase = 're_accumulation' if uptrend else 'accumulation'
        elif price_position < 0.3 and higher_lows and accumulation_volume:
            phase = 're_accumulation' if uptrend else 'accumulation'
        # 분배: 박스 상단 + Lower Highs + 볼륨 조건
        elif upthrust:
            phase = 're_distribution' if downtrend else 'distribution'
        elif price_position > 0.7 and lower_highs and distribution_volume:
            phase = 're_distribution' if downtrend else 'distribution'
        # Markup/Markdown
        elif current_close > box_high:
            phase = 'markup'
        elif current_close < box_low:
            phase = 'markdown'

        return {
            'phase': phase,
            'spring': spring,
            'upthrust': upthrust,
            'box_high': box_high,
            'box_low': box_low,
            'higher_lows': higher_lows,
            'lower_highs': lower_highs,
            'accumulation_volume': accumulation_volume,
            'distribution_volume': distribution_volume
        }

    def _check_reversal_candle(self, df: pd.DataFrame, idx: int) -> Optional[str]:
        """반전 캔들 패턴 체크"""
        if idx < 1:
            return None

        row = df.iloc[idx]
        prev = df.iloc[idx - 1]

        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        total_range = h - l

        if total_range == 0:
            return None

        # 망치형 (Hammer): 긴 아래꼬리, 작은 몸통, 작은 위꼬리
        if lower_wick > body * 2 and upper_wick < body and c > o:
            return 'bullish'

        # 역망치형 (Shooting Star): 긴 위꼬리, 작은 몸통, 작은 아래꼬리
        if upper_wick > body * 2 and lower_wick < body and c < o:
            return 'bearish'

        # 상승장악형 (Bullish Engulfing)
        if prev['close'] < prev['open'] and c > o and c > prev['open'] and o < prev['close']:
            return 'bullish'

        # 하락장악형 (Bearish Engulfing)
        if prev['close'] > prev['open'] and c < o and c < prev['open'] and o > prev['close']:
            return 'bearish'

        return None

    def _check_exit(
        self,
        pos: Position,
        high: float,
        low: float,
        close: float,
        bar: int,
        time: datetime,
        atr: float
    ) -> Optional[Tuple[float, str, float]]:
        """청산 체크"""
        cfg = self.config

        if pos.side == 'long':
            # 청산가
            liq_price = pos.entry_price * (1 - 1/pos.leverage + 0.005)

            if low <= liq_price:
                return liq_price, 'liquidation', -pos.margin

            if low <= pos.sl_price:
                exit_price = pos.sl_price * (1 - cfg.slippage)
                pnl = (exit_price - pos.entry_price) * pos.quantity
                pnl -= pos.entry_price * pos.quantity * cfg.maker_fee
                pnl -= exit_price * pos.quantity * cfg.taker_fee
                return exit_price, 'sl', pnl

            if high >= pos.tp_price:
                exit_price = pos.tp_price
                pnl = (exit_price - pos.entry_price) * pos.quantity
                pnl -= pos.entry_price * pos.quantity * cfg.maker_fee
                pnl -= exit_price * pos.quantity * cfg.maker_fee
                return exit_price, 'tp', pnl

        else:  # short
            liq_price = pos.entry_price * (1 + 1/pos.leverage - 0.005)

            if high >= liq_price:
                return liq_price, 'liquidation', -pos.margin

            if high >= pos.sl_price:
                exit_price = pos.sl_price * (1 + cfg.slippage)
                pnl = (pos.entry_price - exit_price) * pos.quantity
                pnl -= pos.entry_price * pos.quantity * cfg.maker_fee
                pnl -= exit_price * pos.quantity * cfg.taker_fee
                return exit_price, 'sl', pnl

            if low <= pos.tp_price:
                exit_price = pos.tp_price
                pnl = (pos.entry_price - exit_price) * pos.quantity
                pnl -= pos.entry_price * pos.quantity * cfg.maker_fee
                pnl -= exit_price * pos.quantity * cfg.maker_fee
                return exit_price, 'tp', pnl

        # 시간 손절
        if bar - pos.entry_bar >= cfg.max_hold_bars:
            exit_price = close * (1 - cfg.slippage if pos.side == 'long' else 1 + cfg.slippage)
            if pos.side == 'long':
                pnl = (exit_price - pos.entry_price) * pos.quantity
            else:
                pnl = (pos.entry_price - exit_price) * pos.quantity
            pnl -= pos.entry_price * pos.quantity * cfg.maker_fee
            pnl -= exit_price * pos.quantity * cfg.taker_fee
            return exit_price, 'timeout', pnl

        return None

    def _calc_equity(self, cash: float, price: float) -> float:
        """현재 자산 계산"""
        equity = cash
        if self.position:
            if self.position.side == 'long':
                unrealized = (price - self.position.entry_price) * self.position.quantity
            else:
                unrealized = (self.position.entry_price - price) * self.position.quantity
            equity += self.position.margin + unrealized
        return equity

    def _generate_result(self, symbol: str, initial_capital: float) -> Dict[str, Any]:
        """결과 생성"""
        equity_df = pd.DataFrame(self.equity_history)
        if len(equity_df) > 0:
            equity_df = equity_df.set_index('time')

        trades_df = pd.DataFrame(self.trades)
        signals_df = pd.DataFrame(self.signals)

        # 통계
        if len(equity_df) > 0:
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity - initial_capital) / initial_capital * 100
            running_max = equity_df['equity'].cummax()
            drawdown = (equity_df['equity'] - running_max) / running_max * 100
            max_drawdown = drawdown.min()
        else:
            final_equity = initial_capital
            total_return = 0
            max_drawdown = 0

        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            wins = trades_df[trades_df['pnl'] > 0]
            losses = trades_df[trades_df['pnl'] < 0]
            win_rate = len(wins) / len(trades_df) * 100

            total_profit = wins['pnl'].sum() if len(wins) > 0 else 0
            total_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            avg_pnl = trades_df['pnl'].mean()

            long_trades = trades_df[trades_df['side'] == 'long']
            short_trades = trades_df[trades_df['side'] == 'short']
        else:
            win_rate = 0
            profit_factor = 0
            avg_pnl = 0
            long_trades = pd.DataFrame()
            short_trades = pd.DataFrame()

        stats = {
            'symbol': symbol,
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades_df),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_pnl': avg_pnl,
            'total_signals': len(signals_df)
        }

        # 청산 유형별 통계
        if len(trades_df) > 0 and 'exit_reason' in trades_df.columns:
            stats['tp_exits'] = len(trades_df[trades_df['exit_reason'] == 'tp'])
            stats['sl_exits'] = len(trades_df[trades_df['exit_reason'] == 'sl'])
            stats['timeout_exits'] = len(trades_df[trades_df['exit_reason'] == 'timeout'])
            stats['liquidations'] = len(trades_df[trades_df['exit_reason'] == 'liquidation'])

        return {
            'equity_df': equity_df,
            'trades_df': trades_df,
            'signals_df': signals_df,
            'stats': stats
        }
