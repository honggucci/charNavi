"""
선물 백테스트 엔진 V7 - 15분봉 메인 타임프레임
Futures Backtest Engine V7 - 15M Main Timeframe

변경사항:
- 메인 루프: 15분봉 (기존 5분봉 → 15분봉)
- 5분봉: 보조지표로만 사용 (Stoch RSI 확인용)
- 진입/청산: 15분봉 기준

타임프레임 구조:
- 4H: 대추세 필터 (EMA 50/200)
- 1H: Wyckoff 구조
- 15M: 메인 (RSI 다이버전스 + 진입/청산)
- 5M: 보조 (Stoch RSI 과매수/과매도 확인)
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from wpcn.features.indicators import atr, rsi, stoch_rsi, detect_rsi_divergence, detect_fib_bounce


@dataclass
class MTF15Config:
    """15분봉 메인 백테스트 설정"""
    leverage: float = 3.0
    position_pct: float = 0.10

    # ATR 기반 SL/TP
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0

    # 신호 조건
    min_score: float = 4.0
    min_tf_alignment: int = 2
    min_rr_ratio: float = 2.0

    # 시간 설정 (15분봉 기준)
    warmup_bars: int = 500  # 15분봉 500개 = 약 5일
    analysis_interval: int = 1  # 매 봉마다 분석
    cooldown_bars: int = 8  # 2시간 쿨다운 (15분 × 8)
    max_hold_bars: int = 48  # 12시간 최대 보유 (15분 × 48)

    # EMA (4시간봉)
    ema_short: int = 50
    ema_long: int = 200

    # Wyckoff (1시간봉)
    box_lookback: int = 50
    spring_threshold: float = 0.005

    # RSI/Stoch RSI
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
    side: str
    entry_price: float
    quantity: float
    margin: float
    leverage: float
    tp_price: float
    sl_price: float
    entry_bar: int
    entry_time: datetime
    score: float
    reasons: Dict[str, str]


class MTF15BacktestEngine:
    """
    15분봉 메인 MTF 백테스트 엔진

    핵심 변경:
    - 15분봉으로 메인 루프 실행
    - 5분봉은 Stoch RSI 보조 확인만
    """

    def __init__(self, config: MTF15Config = None):
        self.config = config or MTF15Config()
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
        """백테스트 실행 (15분봉 메인)"""
        cfg = self.config

        # 초기화
        self.position = None
        self.trades = []
        self.signals = []
        self.equity_history = []

        cash = initial_capital
        last_entry_bar = -cfg.cooldown_bars - 1

        # 리샘플링 - 15분봉이 메인
        df_15m = self._resample(df_5m, '15min')
        df_1h = self._resample(df_5m, '1h')
        df_4h = self._resample(df_5m, '4h')

        # 4시간봉 EMA
        df_4h['ema_short'] = df_4h['close'].ewm(span=cfg.ema_short).mean()
        df_4h['ema_long'] = df_4h['close'].ewm(span=cfg.ema_long).mean()

        # 15분봉 ATR
        atr_15m = atr(df_15m, 14)

        # 15분봉 RSI 다이버전스
        div_15m = detect_rsi_divergence(df_15m, rsi_len=cfg.rsi_period, lookback=cfg.div_lookback)

        # 15분봉 피보나치
        fib_15m = detect_fib_bounce(df_15m, swing_lookback=100, tolerance=0.005)

        # 15분봉 RSI & Stoch RSI
        df_15m['rsi'] = rsi(df_15m['close'], cfg.rsi_period)
        stoch_k_15m, stoch_d_15m = stoch_rsi(df_15m['close'], cfg.stoch_rsi_period, cfg.stoch_rsi_period, 3, 3)

        # 5분봉 Stoch RSI (보조용)
        stoch_k_5m, stoch_d_5m = stoch_rsi(df_5m['close'], cfg.stoch_rsi_period, cfg.stoch_rsi_period, 3, 3)

        n = len(df_15m)
        idx = df_15m.index

        print(f"Running MTF-15M backtest: {n} bars (15m), {symbol}")

        for i in range(cfg.warmup_bars, n):
            t = idx[i]
            row = df_15m.iloc[i]
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            current_atr = atr_15m.iloc[i] if i < len(atr_15m) and not pd.isna(atr_15m.iloc[i]) else c * 0.01

            # === 1. 포지션 청산 체크 ===
            if self.position:
                exit_result = self._check_exit(self.position, h, l, c, i, t, cfg)
                if exit_result:
                    exit_price, exit_reason, pnl = exit_result
                    cash += self.position.margin + pnl

                    self.trades.append({
                        'symbol': symbol,
                        'side': self.position.side,
                        'entry_price': self.position.entry_price,
                        'exit_price': exit_price,
                        'quantity': self.position.quantity,
                        'pnl': pnl,
                        'pnl_pct': pnl / self.position.margin * 100,
                        'entry_time': self.position.entry_time,
                        'exit_time': t,
                        'exit_reason': exit_reason,
                        'hold_bars': i - self.position.entry_bar,
                        'score': self.position.score,
                        'reasons': self.position.reasons
                    })
                    self.position = None

            # === 2. 신호 분석 ===
            if self.position is None and i - last_entry_bar >= cfg.cooldown_bars:
                signal = self._analyze_signal(
                    df_5m, df_15m, df_1h, df_4h,
                    div_15m, fib_15m,
                    stoch_k_15m, stoch_d_15m,
                    stoch_k_5m, stoch_d_5m,
                    i, t, current_atr, c
                )

                if signal and signal['action'] != 'none' and cash > 0:
                    self.signals.append(signal)

                    # 진입
                    direction = signal['action']
                    entry_price = c * (1 + cfg.slippage if direction == 'long' else 1 - cfg.slippage)

                    if direction == 'long':
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
                        position_value = cash * cfg.position_pct * cfg.leverage
                        quantity = position_value / entry_price
                        margin = position_value / cfg.leverage
                        fee = position_value * cfg.maker_fee

                        self.position = Position(
                            symbol=symbol,
                            side=direction,
                            entry_price=entry_price,
                            quantity=quantity,
                            margin=margin,
                            leverage=cfg.leverage,
                            tp_price=tp_price,
                            sl_price=sl_price,
                            entry_bar=i,
                            entry_time=t,
                            score=signal['score'],
                            reasons=signal['reasons']
                        )

                        cash -= margin + fee
                        last_entry_bar = i

            # === Equity 기록 ===
            equity = self._calc_equity(cash, c)
            self.equity_history.append({
                'time': t,
                'equity': equity,
                'cash': cash
            })

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

    def _analyze_signal(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        div_15m: pd.DataFrame,
        fib_15m: pd.DataFrame,
        stoch_k_15m: pd.Series,
        stoch_d_15m: pd.Series,
        stoch_k_5m: pd.Series,
        stoch_d_5m: pd.Series,
        bar_idx: int,
        timestamp: datetime,
        atr_val: float,
        current_price: float
    ) -> Optional[Dict]:
        """신호 분석"""
        cfg = self.config

        long_score = 0.0
        short_score = 0.0
        reasons = {}
        tf_alignment = 0

        # 완료된 이전 봉 타임스탬프
        t_15m_prev = timestamp - timedelta(minutes=15)
        t_1h_prev = timestamp.floor('1h') - timedelta(hours=1)
        t_4h_prev = timestamp.floor('4h') - timedelta(hours=4)

        # === 4H 추세 필터 ===
        if t_4h_prev in df_4h.index:
            row_4h = df_4h.loc[t_4h_prev]
            close_4h = row_4h['close']
            ema_short = row_4h.get('ema_short')
            ema_long = row_4h.get('ema_long')

            if not pd.isna(ema_short) and not pd.isna(ema_long):
                if close_4h > ema_short > ema_long:
                    long_score += 2.0
                    short_score -= 2.0
                    reasons['4h_trend'] = 'BULLISH'
                    tf_alignment += 1
                elif close_4h < ema_short < ema_long:
                    short_score += 2.0
                    long_score -= 2.0
                    reasons['4h_trend'] = 'BEARISH'
                    tf_alignment += 1
                else:
                    reasons['4h_trend'] = 'NEUTRAL'

        # === 1H Wyckoff ===
        if t_1h_prev in df_1h.index:
            wyckoff = self._analyze_wyckoff(df_1h, t_1h_prev, cfg)
            reasons['1h_wyckoff'] = wyckoff['phase'].upper()

            if wyckoff['phase'] in ['accumulation', 're_accumulation']:
                long_score += 2.0
                tf_alignment += 1
            elif wyckoff['phase'] in ['distribution', 're_distribution']:
                short_score += 2.0
                tf_alignment += 1
            elif wyckoff['phase'] == 'markup':
                long_score += 1.0
            elif wyckoff['phase'] == 'markdown':
                short_score += 1.0

            if wyckoff.get('spring'):
                long_score += 1.5
                reasons['1h_event'] = 'SPRING'
            elif wyckoff.get('upthrust'):
                short_score += 1.5
                reasons['1h_event'] = 'UPTHRUST'

        # === 15M RSI 다이버전스 (메인 신호) ===
        if t_15m_prev in div_15m.index:
            div_row = div_15m.loc[t_15m_prev]

            if div_row.get('bullish_div', False):
                strength = div_row.get('div_strength', 0.7)
                long_score += 2.5 * min(strength, 1.0)
                reasons['15m_div'] = f'BULLISH ({strength:.2f})'
                tf_alignment += 1
            elif div_row.get('bearish_div', False):
                strength = div_row.get('div_strength', 0.7)
                short_score += 2.5 * min(strength, 1.0)
                reasons['15m_div'] = f'BEARISH ({strength:.2f})'
                tf_alignment += 1

        # === 15M Stoch RSI (메인 타이밍) ===
        if bar_idx < len(stoch_k_15m) and bar_idx < len(stoch_d_15m):
            k_15m = stoch_k_15m.iloc[bar_idx]
            d_15m = stoch_d_15m.iloc[bar_idx]

            if not pd.isna(k_15m) and not pd.isna(d_15m):
                if k_15m < 20:
                    long_score += 1.5
                    reasons['15m_stoch'] = f'OVERSOLD ({k_15m:.1f})'
                    tf_alignment += 1
                elif k_15m > 80:
                    short_score += 1.5
                    reasons['15m_stoch'] = f'OVERBOUGHT ({k_15m:.1f})'
                    tf_alignment += 1

                # K/D 크로스
                if bar_idx > 0:
                    k_prev = stoch_k_15m.iloc[bar_idx - 1]
                    d_prev = stoch_d_15m.iloc[bar_idx - 1]
                    if not pd.isna(k_prev) and not pd.isna(d_prev):
                        if k_prev < d_prev and k_15m > d_15m and k_15m < 30:
                            long_score += 1.0
                            reasons['15m_cross'] = 'GOLDEN'
                        elif k_prev > d_prev and k_15m < d_15m and k_15m > 70:
                            short_score += 1.0
                            reasons['15m_cross'] = 'DEATH'

        # === 15M 피보나치 ===
        if t_15m_prev in fib_15m.index:
            fib_row = fib_15m.loc[t_15m_prev]
            if fib_row.get('fib_bounce_long', False):
                long_score += 1.0
                reasons['15m_fib'] = 'BOUNCE_LONG'
            elif fib_row.get('fib_bounce_short', False):
                short_score += 1.0
                reasons['15m_fib'] = 'BOUNCE_SHORT'

        # === 5M Stoch RSI (보조 확인) ===
        # 15분봉 타임스탬프에 해당하는 5분봉 인덱스 찾기
        try:
            idx_5m = df_5m.index.get_indexer([timestamp], method='ffill')[0]
            if idx_5m >= 0 and idx_5m < len(stoch_k_5m):
                k_5m = stoch_k_5m.iloc[idx_5m]

                if not pd.isna(k_5m):
                    # 5분봉 Stoch RSI가 같은 방향일 때 보너스
                    if k_5m < 30:
                        long_score += 0.5
                        reasons['5m_stoch'] = f'OVERSOLD ({k_5m:.1f})'
                    elif k_5m > 70:
                        short_score += 0.5
                        reasons['5m_stoch'] = f'OVERBOUGHT ({k_5m:.1f})'
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"5M Stoch RSI failed: ts={timestamp}, error={e}")

        # === 15M 반전 캔들 ===
        candle = self._check_reversal_candle(df_15m, bar_idx)
        if candle == 'bullish':
            long_score += 1.5
            reasons['15m_candle'] = 'HAMMER'
        elif candle == 'bearish':
            short_score += 1.5
            reasons['15m_candle'] = 'SHOOTING_STAR'

        # === 신호 결정 ===
        action = 'none'
        score = 0.0

        # 방향 일관성 체크
        long_consistent = (
            reasons.get('4h_trend') == 'BULLISH' and
            reasons.get('1h_wyckoff') in ['ACCUMULATION', 'RE_ACCUMULATION', 'MARKUP', 'RANGING']
        )
        short_consistent = (
            reasons.get('4h_trend') == 'BEARISH' and
            reasons.get('1h_wyckoff') in ['DISTRIBUTION', 'RE_DISTRIBUTION', 'MARKDOWN', 'RANGING']
        )

        if long_score >= cfg.min_score and long_score > short_score + 1.0:
            if tf_alignment >= cfg.min_tf_alignment and long_consistent:
                action = 'long'
                score = long_score
        elif short_score >= cfg.min_score and short_score > long_score + 1.0:
            if tf_alignment >= cfg.min_tf_alignment and short_consistent:
                action = 'short'
                score = short_score

        return {
            'timestamp': timestamp,
            'action': action,
            'score': score,
            'long_score': long_score,
            'short_score': short_score,
            'tf_alignment': tf_alignment,
            'reasons': reasons
        }

    def _analyze_wyckoff(self, df_1h: pd.DataFrame, timestamp, cfg: MTF15Config) -> Dict:
        """Wyckoff 분석"""
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

        if box_range <= 0:
            return {'phase': 'ranging'}

        # Higher Lows / Lower Highs
        recent_lows = window['low'].rolling(5).min().dropna()
        recent_highs = window['high'].rolling(5).max().dropna()

        higher_lows = len(recent_lows) >= 3 and recent_lows.iloc[-1] > recent_lows.iloc[-3]
        lower_highs = len(recent_highs) >= 3 and recent_highs.iloc[-1] < recent_highs.iloc[-3]

        price_position = (current_close - box_low) / box_range

        # 볼륨 분석
        if 'volume' in window.columns:
            avg_volume = window['volume'].mean()
            down_bars = window[window['close'] < window['open']]
            up_bars = window[window['close'] >= window['open']]
            down_volume = down_bars['volume'].mean() if len(down_bars) > 0 else 0
            up_volume = up_bars['volume'].mean() if len(up_bars) > 0 else 0
            accumulation_volume = down_volume <= up_volume if up_volume > 0 else False
            distribution_volume = up_volume <= down_volume if down_volume > 0 else False
        else:
            avg_volume = 1
            accumulation_volume = False
            distribution_volume = False

        # 스프링/업쓰러스트
        spring = False
        upthrust = False

        recent_low = window['low'].iloc[-3:].min()
        if recent_low < box_low * (1 - cfg.spring_threshold) and current_close > box_low:
            if 'volume' in window.columns:
                if window['volume'].iloc[-3:].max() > avg_volume * 1.3:
                    spring = True
            else:
                spring = True

        recent_high = window['high'].iloc[-3:].max()
        if recent_high > box_high * (1 + cfg.spring_threshold) and current_close < box_high:
            if 'volume' in window.columns:
                if window['volume'].iloc[-3:].max() > avg_volume * 1.3:
                    upthrust = True
            else:
                upthrust = True

        # EMA 추세
        ema_20 = window['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = window['close'].ewm(span=50).mean().iloc[-1] if len(window) >= 50 else ema_20
        uptrend = current_close > ema_20 > ema_50
        downtrend = current_close < ema_20 < ema_50

        # 페이즈 결정
        phase = 'ranging'

        if spring:
            phase = 're_accumulation' if uptrend else 'accumulation'
        elif price_position < 0.3 and higher_lows and accumulation_volume:
            phase = 're_accumulation' if uptrend else 'accumulation'
        elif upthrust:
            phase = 're_distribution' if downtrend else 'distribution'
        elif price_position > 0.7 and lower_highs and distribution_volume:
            phase = 're_distribution' if downtrend else 'distribution'
        elif current_close > box_high:
            phase = 'markup'
        elif current_close < box_low:
            phase = 'markdown'

        return {
            'phase': phase,
            'spring': spring,
            'upthrust': upthrust
        }

    def _check_reversal_candle(self, df: pd.DataFrame, idx: int) -> Optional[str]:
        """반전 캔들 체크"""
        if idx < 1:
            return None

        row = df.iloc[idx]
        prev = df.iloc[idx - 1]

        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        if body == 0:
            return None

        # 망치형
        if lower_wick > body * 2 and upper_wick < body and c > o:
            return 'bullish'

        # 역망치형
        if upper_wick > body * 2 and lower_wick < body and c < o:
            return 'bearish'

        # 장악형
        if prev['close'] < prev['open'] and c > o and c > prev['open'] and o < prev['close']:
            return 'bullish'
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
        cfg: MTF15Config
    ) -> Optional[Tuple[float, str, float]]:
        """청산 체크"""
        if pos.side == 'long':
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

        else:
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
        """현재 자산"""
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
