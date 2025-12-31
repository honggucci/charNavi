"""
선물 백테스트 엔진 V6 - 물리학 공식 통합 버전
Futures Backtest Engine V6 - Physics Integration

핵심 전략:
- 4H EMA 추세 + 1H Wyckoff + 15M RSI 다이버전스 (V4 기반)
- 물리학 검증: 모멘텀, 에너지, 엔트로피, 평균회귀력, 사이클
- 블랙숄즈 + 볼츠만 확률 기반 신호 필터링
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import norm


@dataclass
class PhysicsConfig:
    """물리학 통합 백테스트 설정"""
    leverage: float = 3.0
    position_pct: float = 0.10

    # ATR 기반 TP/SL
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0

    # 신호 필터
    min_score: float = 5.0
    min_tf_alignment: int = 3
    min_rr_ratio: float = 2.0
    min_physics_confidence: float = 0.55  # 물리학 신뢰도 최소값

    # 시간 설정 (5분봉 기준)
    warmup_bars: int = 1000
    analysis_interval: int = 48  # 4시간마다
    cooldown_bars: int = 144  # 12시간
    max_hold_bars: int = 288  # 24시간

    # EMA
    ema_short: int = 50
    ema_long: int = 200

    # Wyckoff
    box_lookback: int = 50
    spring_threshold: float = 0.005

    # 다이버전스
    rsi_period: int = 14
    stoch_rsi_period: int = 14
    div_lookback: int = 20

    # 물리학 파라미터
    momentum_window: int = 14
    energy_window: int = 20
    entropy_window: int = 20
    fft_window: int = 64
    mean_reversion_lookback: int = 50

    # 비용
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage: float = 0.0001


@dataclass
class Position:
    """포지션"""
    side: str
    entry_price: float
    quantity: float
    margin: float
    leverage: float
    tp_price: float
    sl_price: float
    entry_bar: int
    entry_time: datetime


class PhysicsBacktestEngine:
    """
    물리학 공식 통합 백테스트 엔진

    물리학 검증 요소:
    1. 모멘텀 (F = ma): 가격 가속도 분석
    2. 에너지 (E = ½mv²): 거래량 × 변동폭
    3. 엔트로피: 시장 혼란도/질서도
    4. 평균회귀력 (F = -kx): 스프링 법칙
    5. 사이클 타이밍: FFT 기반
    6. 블랙숄즈/볼츠만 확률
    """

    def __init__(self, config: PhysicsConfig = None):
        self.config = config or PhysicsConfig()
        self.position: Optional[Position] = None
        self.trades: List[Dict] = []
        self.equity_history: List[Dict] = []
        self.signals: List[Dict] = []

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
        self.equity_history = []
        self.signals = []

        cash = initial_capital
        last_entry_bar = -cfg.cooldown_bars - 1
        cached_signal = None

        # 리샘플링
        df_15m = self._resample(df_5m, '15min')
        df_1h = self._resample(df_5m, '1h')
        df_4h = self._resample(df_5m, '4h')

        # 지표 계산
        self._calculate_indicators(df_5m, df_15m, df_1h, df_4h)

        n = len(df_5m)
        idx = df_5m.index

        print(f"Running Physics V6 backtest: {n} bars, {symbol}")

        for i in range(cfg.warmup_bars, n):
            t = idx[i]
            row = df_5m.iloc[i]
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            current_atr = row['atr'] if not pd.isna(row['atr']) else c * 0.01

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
                        'hold_bars': i - self.position.entry_bar
                    })
                    self.position = None

            # === 2. 분석 주기에 신호 체크 ===
            if i % cfg.analysis_interval == 0:
                signal = self._analyze_signal(
                    df_5m, df_15m, df_1h, df_4h, t, i, current_atr, c
                )
                if signal:
                    cached_signal = signal
                    self.signals.append({
                        'time': t,
                        'signal': signal['direction'],
                        'score': signal['score'],
                        'physics_confidence': signal['physics_confidence'],
                        'components': signal['components']
                    })

            # === 3. 진입 체크 ===
            if (self.position is None and
                cached_signal and
                i - last_entry_bar >= cfg.cooldown_bars and
                cash > 0):

                direction = cached_signal['direction']

                # 슬리피지 적용
                if direction == 'long':
                    entry_price = c * (1 + cfg.slippage)
                    sl_price = entry_price - current_atr * cfg.sl_atr_mult
                    tp_price = entry_price + current_atr * cfg.tp_atr_mult
                else:
                    entry_price = c * (1 - cfg.slippage)
                    sl_price = entry_price + current_atr * cfg.sl_atr_mult
                    tp_price = entry_price - current_atr * cfg.tp_atr_mult

                # 손익비 확인
                risk = abs(entry_price - sl_price)
                reward = abs(tp_price - entry_price)
                rr_ratio = reward / risk if risk > 0 else 0

                if rr_ratio >= cfg.min_rr_ratio:
                    position_value = cash * cfg.position_pct * cfg.leverage
                    quantity = position_value / entry_price
                    margin = position_value / cfg.leverage
                    fee = position_value * cfg.maker_fee

                    self.position = Position(
                        side=direction,
                        entry_price=entry_price,
                        quantity=quantity,
                        margin=margin,
                        leverage=cfg.leverage,
                        tp_price=tp_price,
                        sl_price=sl_price,
                        entry_bar=i,
                        entry_time=t
                    )

                    cash -= margin + fee
                    last_entry_bar = i
                    cached_signal = None

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

    def _calculate_indicators(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame
    ):
        """모든 지표 계산"""
        cfg = self.config

        # ATR (5분봉)
        df_5m['tr'] = np.maximum(
            df_5m['high'] - df_5m['low'],
            np.maximum(
                abs(df_5m['high'] - df_5m['close'].shift(1)),
                abs(df_5m['low'] - df_5m['close'].shift(1))
            )
        )
        df_5m['atr'] = df_5m['tr'].rolling(14).mean()

        # 물리학 지표 (5분봉)
        self._calculate_physics_indicators(df_5m)

        # 4시간봉 EMA
        df_4h['ema_short'] = df_4h['close'].ewm(span=cfg.ema_short).mean()
        df_4h['ema_long'] = df_4h['close'].ewm(span=cfg.ema_long).mean()

        # 1시간봉 Wyckoff
        df_1h['box_high'] = df_1h['high'].rolling(cfg.box_lookback).max()
        df_1h['box_low'] = df_1h['low'].rolling(cfg.box_lookback).min()

        # 15분봉 RSI & Stoch RSI
        delta = df_15m['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=cfg.rsi_period).mean()
        avg_loss = loss.ewm(span=cfg.rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df_15m['rsi'] = 100 - (100 / (1 + rs))

        # Stoch RSI
        rsi_low = df_15m['rsi'].rolling(cfg.stoch_rsi_period).min()
        rsi_high = df_15m['rsi'].rolling(cfg.stoch_rsi_period).max()
        df_15m['stoch_rsi'] = (df_15m['rsi'] - rsi_low) / (rsi_high - rsi_low + 1e-10) * 100

        # 5분봉 Stoch RSI
        delta_5m = df_5m['close'].diff()
        gain_5m = delta_5m.where(delta_5m > 0, 0.0)
        loss_5m = -delta_5m.where(delta_5m < 0, 0.0)
        avg_gain_5m = gain_5m.ewm(span=cfg.rsi_period).mean()
        avg_loss_5m = loss_5m.ewm(span=cfg.rsi_period).mean()
        rs_5m = avg_gain_5m / (avg_loss_5m + 1e-10)
        df_5m['rsi'] = 100 - (100 / (1 + rs_5m))

        rsi_low_5m = df_5m['rsi'].rolling(cfg.stoch_rsi_period).min()
        rsi_high_5m = df_5m['rsi'].rolling(cfg.stoch_rsi_period).max()
        df_5m['stoch_rsi_k'] = (df_5m['rsi'] - rsi_low_5m) / (rsi_high_5m - rsi_low_5m + 1e-10) * 100
        df_5m['stoch_rsi_d'] = df_5m['stoch_rsi_k'].rolling(3).mean()

    def _calculate_physics_indicators(self, df: pd.DataFrame):
        """물리학 지표 계산"""
        cfg = self.config
        close = df['close']

        # 1. 모멘텀 (가속도)
        velocity = close.pct_change()
        acceleration = velocity.diff()
        df['momentum'] = acceleration.rolling(cfg.momentum_window).mean()

        # 2. 에너지 (거래량 × 변동폭²)
        volume = df['volume']
        price_range = df['high'] - df['low']
        vol_norm = volume / (volume.rolling(cfg.energy_window).mean() + 1e-10)
        range_norm = price_range / (price_range.rolling(cfg.energy_window).mean() + 1e-10)
        df['energy'] = (vol_norm * (range_norm ** 2)).rolling(5).mean()

        # 3. 엔트로피 (Shannon Entropy)
        returns = close.pct_change()
        entropy_values = []
        for i in range(len(df)):
            if i < cfg.entropy_window:
                entropy_values.append(np.nan)
                continue
            window_returns = returns.iloc[max(0, i - cfg.entropy_window):i].dropna()
            if len(window_returns) < cfg.entropy_window // 2:
                entropy_values.append(np.nan)
                continue
            hist, _ = np.histogram(window_returns, bins=10, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            entropy_values.append(entropy)
        df['entropy'] = entropy_values

        # 4. 평균 회귀력 (스프링 법칙: F = -k × (x - x_mean))
        sma = close.rolling(cfg.mean_reversion_lookback).mean()
        std = close.rolling(cfg.mean_reversion_lookback).std()
        df['mean_reversion'] = -(close - sma) / (std + 1e-10)

        # 5. 추세 관성
        changes = close.diff()
        inertia_values = []
        for i in range(len(df)):
            if i < 20:
                inertia_values.append(np.nan)
                continue
            window_changes = changes.iloc[i-20:i]
            positive_ratio = (window_changes > 0).sum() / 20
            negative_ratio = (window_changes < 0).sum() / 20
            inertia = abs(positive_ratio - negative_ratio) * 2
            if positive_ratio < negative_ratio:
                inertia = -inertia
            inertia_values.append(inertia)
        df['trend_inertia'] = inertia_values

    def _analyze_signal(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        timestamp: datetime,
        bar_idx: int,
        atr: float,
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """신호 분석 (물리학 검증 포함)"""
        cfg = self.config

        # 완료된 이전 봉 타임스탬프
        t_5m_prev = timestamp.floor('5min') - timedelta(minutes=5)
        t_15m_prev = timestamp.floor('15min') - timedelta(minutes=15)
        t_1h_prev = timestamp.floor('1h') - timedelta(hours=1)
        t_4h_prev = timestamp.floor('4h') - timedelta(hours=4)

        long_score = 0.0
        short_score = 0.0
        components = {}
        tf_alignment_long = 0
        tf_alignment_short = 0

        # === 1. 4H 추세 분석 ===
        if t_4h_prev in df_4h.index:
            row_4h = df_4h.loc[t_4h_prev]
            ema_short = row_4h.get('ema_short')
            ema_long = row_4h.get('ema_long')
            close_4h = row_4h.get('close')

            if not pd.isna(ema_short) and not pd.isna(ema_long) and not pd.isna(close_4h):
                if close_4h > ema_short > ema_long:
                    long_score += 2.0
                    tf_alignment_long += 1
                    components['4h_trend'] = 'BULLISH'
                elif close_4h < ema_short < ema_long:
                    short_score += 2.0
                    tf_alignment_short += 1
                    components['4h_trend'] = 'BEARISH'
                else:
                    components['4h_trend'] = 'NEUTRAL'

        # === 2. 1H Wyckoff 분석 (볼륨 포함) ===
        if t_1h_prev in df_1h.index:
            wyckoff = self._analyze_wyckoff(df_1h, t_1h_prev, cfg)
            components['1h_wyckoff'] = wyckoff['phase']

            if wyckoff['phase'] in ['ACCUMULATION', 'RE_ACCUMULATION']:
                long_score += 2.0
                tf_alignment_long += 1
            elif wyckoff['phase'] in ['DISTRIBUTION', 'RE_DISTRIBUTION']:
                short_score += 2.0
                tf_alignment_short += 1

            if wyckoff.get('spring'):
                long_score += 1.5
                components['1h_spring'] = True
            if wyckoff.get('upthrust'):
                short_score += 1.5
                components['1h_upthrust'] = True

        # === 3. 15M RSI 다이버전스 ===
        if t_15m_prev in df_15m.index:
            divergence = self._analyze_divergence(df_15m, t_15m_prev, cfg)

            if divergence['bullish']:
                long_score += 2.0 * divergence['strength']
                tf_alignment_long += 1
                components['15m_divergence'] = 'BULLISH'
            elif divergence['bearish']:
                short_score += 2.0 * divergence['strength']
                tf_alignment_short += 1
                components['15m_divergence'] = 'BEARISH'

        # === 4. 5M 진입 타이밍 ===
        if t_5m_prev in df_5m.index:
            row_5m = df_5m.loc[t_5m_prev]
            stoch_k = row_5m.get('stoch_rsi_k')
            stoch_d = row_5m.get('stoch_rsi_d')

            if not pd.isna(stoch_k) and not pd.isna(stoch_d):
                if stoch_k < 20:
                    long_score += 1.0
                    tf_alignment_long += 1
                    components['5m_stoch'] = 'OVERSOLD'
                elif stoch_k > 80:
                    short_score += 1.0
                    tf_alignment_short += 1
                    components['5m_stoch'] = 'OVERBOUGHT'

        # === 5. 물리학 검증 ===
        physics_result = self._analyze_physics(df_5m, bar_idx, current_price, atr)
        components['physics'] = physics_result

        # 물리학 신뢰도 기반 점수 조정
        physics_conf = physics_result['confidence']

        if physics_result['direction'] == 'up':
            long_score *= (1 + physics_conf * 0.3)
            short_score *= (1 - physics_conf * 0.2)
        elif physics_result['direction'] == 'down':
            short_score *= (1 + physics_conf * 0.3)
            long_score *= (1 - physics_conf * 0.2)

        # === 신호 결정 ===
        final_direction = None
        final_score = 0
        final_alignment = 0

        if long_score >= cfg.min_score and long_score > short_score:
            if tf_alignment_long >= cfg.min_tf_alignment:
                # 방향 일관성 체크
                if components.get('4h_trend') in ['BULLISH', None]:
                    if physics_conf >= cfg.min_physics_confidence or physics_result['direction'] != 'down':
                        final_direction = 'long'
                        final_score = long_score
                        final_alignment = tf_alignment_long

        elif short_score >= cfg.min_score and short_score > long_score:
            if tf_alignment_short >= cfg.min_tf_alignment:
                if components.get('4h_trend') in ['BEARISH', None]:
                    if physics_conf >= cfg.min_physics_confidence or physics_result['direction'] != 'up':
                        final_direction = 'short'
                        final_score = short_score
                        final_alignment = tf_alignment_short

        if final_direction:
            return {
                'direction': final_direction,
                'score': final_score,
                'tf_alignment': final_alignment,
                'physics_confidence': physics_conf,
                'components': components
            }

        return None

    def _analyze_physics(
        self,
        df_5m: pd.DataFrame,
        bar_idx: int,
        current_price: float,
        atr: float
    ) -> Dict[str, Any]:
        """물리학 지표 분석"""
        cfg = self.config

        # 현재 물리학 지표 값
        momentum = df_5m['momentum'].iloc[bar_idx] if bar_idx < len(df_5m) else 0
        energy = df_5m['energy'].iloc[bar_idx] if bar_idx < len(df_5m) else 1
        entropy = df_5m['entropy'].iloc[bar_idx] if bar_idx < len(df_5m) else 2.5
        mean_reversion = df_5m['mean_reversion'].iloc[bar_idx] if bar_idx < len(df_5m) else 0
        trend_inertia = df_5m['trend_inertia'].iloc[bar_idx] if bar_idx < len(df_5m) else 0

        # NaN 처리
        momentum = float(momentum) if not pd.isna(momentum) else 0
        energy = float(energy) if not pd.isna(energy) else 1
        entropy = float(entropy) if not pd.isna(entropy) else 2.5
        mean_reversion = float(mean_reversion) if not pd.isna(mean_reversion) else 0
        trend_inertia = float(trend_inertia) if not pd.isna(trend_inertia) else 0

        # FFT 사이클 분석
        fft_result = self._analyze_fft_cycle(df_5m, bar_idx)

        # 방향 판단
        up_score = 0.0
        down_score = 0.0

        # 1. 모멘텀 (가속도)
        if momentum > 0.0001:
            up_score += 0.2
        elif momentum < -0.0001:
            down_score += 0.2

        # 2. 에너지 (높은 에너지 = 큰 움직임 가능)
        if energy > 2.0:
            # 에너지 높으면 방향성 강화
            if trend_inertia > 0:
                up_score += 0.1
            elif trend_inertia < 0:
                down_score += 0.1

        # 3. 엔트로피 (낮을수록 질서정연 = 추세 신뢰)
        entropy_factor = 1.0
        if entropy < 2.0:
            entropy_factor = 1.2  # 질서정연하면 신호 강화
        elif entropy > 3.5:
            entropy_factor = 0.8  # 혼란스러우면 신호 약화

        # 4. 평균 회귀력 (스프링)
        if mean_reversion > 1.5:
            up_score += 0.25  # 강한 하방 이탈 → 복귀 예상
        elif mean_reversion < -1.5:
            down_score += 0.25  # 강한 상방 이탈 → 복귀 예상
        elif mean_reversion > 0.5:
            up_score += 0.1
        elif mean_reversion < -0.5:
            down_score += 0.1

        # 5. 추세 관성
        if trend_inertia > 0.4:
            up_score += 0.2 * abs(trend_inertia)
        elif trend_inertia < -0.4:
            down_score += 0.2 * abs(trend_inertia)

        # 6. FFT 사이클 타이밍
        phase = fft_result['current_phase']
        if 0.6 < phase < 0.9:  # 사이클 저점 근처
            up_score += 0.15
        elif 0.1 < phase < 0.4:  # 사이클 고점 근처
            down_score += 0.15

        # 엔트로피 팩터 적용
        up_score *= entropy_factor
        down_score *= entropy_factor

        # 방향 결정
        if up_score > down_score + 0.1:
            direction = 'up'
            confidence = min(0.95, up_score / (up_score + down_score + 0.01))
        elif down_score > up_score + 0.1:
            direction = 'down'
            confidence = min(0.95, down_score / (up_score + down_score + 0.01))
        else:
            direction = 'neutral'
            confidence = 0.5

        return {
            'direction': direction,
            'confidence': confidence,
            'momentum': momentum,
            'energy': energy,
            'entropy': entropy,
            'mean_reversion': mean_reversion,
            'trend_inertia': trend_inertia,
            'fft_phase': phase,
            'fft_cycle': fft_result['dominant_cycle']
        }

    def _analyze_fft_cycle(self, df: pd.DataFrame, bar_idx: int) -> Dict[str, Any]:
        """FFT 사이클 분석"""
        cfg = self.config

        start_idx = max(0, bar_idx - cfg.fft_window)
        close = df['close'].iloc[start_idx:bar_idx].values

        if len(close) < cfg.fft_window // 2:
            return {'dominant_cycle': 20, 'current_phase': 0.5, 'amplitude': 0}

        # 디트렌딩
        x = np.arange(len(close))
        coeffs = np.polyfit(x, close, 1)
        trend = np.polyval(coeffs, x)
        detrended = close - trend

        # FFT
        fft_values = fft(detrended)
        frequencies = fftfreq(len(detrended), 1)

        positive_mask = frequencies > 0
        fft_magnitude = np.abs(fft_values[positive_mask])
        positive_freqs = frequencies[positive_mask]

        if len(fft_magnitude) == 0:
            return {'dominant_cycle': 20, 'current_phase': 0.5, 'amplitude': 0}

        # 지배적 주파수
        dominant_idx = np.argmax(fft_magnitude)
        dominant_freq = positive_freqs[dominant_idx]

        dominant_cycle = int(1 / dominant_freq) if dominant_freq > 0 else 20
        dominant_cycle = max(5, min(100, dominant_cycle))

        # 위상
        phase = np.angle(fft_values[positive_mask][dominant_idx])
        current_phase = (phase + np.pi) / (2 * np.pi)

        # 진폭
        amplitude = fft_magnitude[dominant_idx] / len(close)

        return {
            'dominant_cycle': dominant_cycle,
            'current_phase': current_phase,
            'amplitude': amplitude
        }

    def _analyze_wyckoff(
        self,
        df_1h: pd.DataFrame,
        timestamp: datetime,
        cfg: PhysicsConfig
    ) -> Dict[str, Any]:
        """Wyckoff 분석 (볼륨 포함)"""
        result = {
            'phase': 'RANGING',
            'spring': False,
            'upthrust': False
        }

        if timestamp not in df_1h.index:
            return result

        idx = df_1h.index.get_loc(timestamp)
        if idx < cfg.box_lookback:
            return result

        window = df_1h.iloc[idx - cfg.box_lookback:idx + 1]

        box_high = window['high'].max()
        box_low = window['low'].min()
        box_range = box_high - box_low

        if box_range <= 0:
            return result

        current = df_1h.iloc[idx]
        close = current['close']
        low = current['low']
        high = current['high']
        volume = current.get('volume', 0)

        avg_volume = window['volume'].mean() if 'volume' in window.columns else 1

        # 볼륨 분석
        down_bars = window[window['close'] < window['open']]
        up_bars = window[window['close'] >= window['open']]
        down_volume = down_bars['volume'].mean() if len(down_bars) > 0 else 0
        up_volume = up_bars['volume'].mean() if len(up_bars) > 0 else 0

        # 박스 내 위치
        position_in_box = (close - box_low) / box_range

        # 스프링 감지
        spring_threshold = box_low - box_range * cfg.spring_threshold
        if low < spring_threshold and close > box_low:
            if volume > avg_volume * 1.5:  # 볼륨 서지 필요
                result['spring'] = True

        # 업쓰러스트 감지
        upthrust_threshold = box_high + box_range * cfg.spring_threshold
        if high > upthrust_threshold and close < box_high:
            if volume > avg_volume * 1.5:
                result['upthrust'] = True

        # 페이즈 판단
        if position_in_box < 0.3:
            if down_volume <= up_volume:  # 하락에 볼륨 감소 = 매집
                result['phase'] = 'ACCUMULATION'
        elif position_in_box > 0.7:
            if up_volume <= down_volume:  # 상승에 볼륨 감소 = 분산
                result['phase'] = 'DISTRIBUTION'
        else:
            # 추세 중 패턴
            higher_lows = self._check_higher_lows(window)
            lower_highs = self._check_lower_highs(window)

            if higher_lows and down_volume < up_volume * 0.8:
                result['phase'] = 'RE_ACCUMULATION'
            elif lower_highs and up_volume < down_volume * 0.8:
                result['phase'] = 'RE_DISTRIBUTION'

        return result

    def _check_higher_lows(self, df: pd.DataFrame) -> bool:
        """Higher Lows 패턴 체크"""
        lows = df['low'].values
        if len(lows) < 10:
            return False

        # 최근 10봉 중 저점 3개 체크
        recent_lows = []
        for i in range(2, len(lows) - 2):
            if lows[i] == min(lows[i-2:i+3]):
                recent_lows.append(lows[i])

        if len(recent_lows) >= 2:
            return recent_lows[-1] > recent_lows[0]
        return False

    def _check_lower_highs(self, df: pd.DataFrame) -> bool:
        """Lower Highs 패턴 체크"""
        highs = df['high'].values
        if len(highs) < 10:
            return False

        recent_highs = []
        for i in range(2, len(highs) - 2):
            if highs[i] == max(highs[i-2:i+3]):
                recent_highs.append(highs[i])

        if len(recent_highs) >= 2:
            return recent_highs[-1] < recent_highs[0]
        return False

    def _analyze_divergence(
        self,
        df_15m: pd.DataFrame,
        timestamp: datetime,
        cfg: PhysicsConfig
    ) -> Dict[str, Any]:
        """RSI 다이버전스 분석"""
        result = {'bullish': False, 'bearish': False, 'strength': 0.0}

        if timestamp not in df_15m.index:
            return result

        idx = df_15m.index.get_loc(timestamp)
        if idx < cfg.div_lookback:
            return result

        window = df_15m.iloc[idx - cfg.div_lookback:idx + 1]

        close = window['close']
        rsi = window['rsi']
        stoch_rsi = window.get('stoch_rsi')

        if stoch_rsi is None or len(stoch_rsi) == 0:
            return result

        current_stoch = stoch_rsi.iloc[-1]
        if pd.isna(current_stoch):
            return result

        # 과매도에서만 Bullish 체크
        if current_stoch < 30:
            # 가격 저점들 찾기
            price_lows = []
            rsi_lows = []

            for i in range(2, len(close) - 2):
                if close.iloc[i] == close.iloc[i-2:i+3].min():
                    price_lows.append((i, close.iloc[i], rsi.iloc[i]))

            if len(price_lows) >= 2:
                prev = price_lows[-2]
                curr = price_lows[-1]

                # Bullish: 가격 LL + RSI HL
                if curr[1] < prev[1] and curr[2] > prev[2]:
                    strength = min(1.0, abs(curr[2] - prev[2]) / 20)
                    result['bullish'] = True
                    result['strength'] = strength

        # 과매수에서만 Bearish 체크
        elif current_stoch > 70:
            price_highs = []
            rsi_highs = []

            for i in range(2, len(close) - 2):
                if close.iloc[i] == close.iloc[i-2:i+3].max():
                    price_highs.append((i, close.iloc[i], rsi.iloc[i]))

            if len(price_highs) >= 2:
                prev = price_highs[-2]
                curr = price_highs[-1]

                # Bearish: 가격 HH + RSI LH
                if curr[1] > prev[1] and curr[2] < prev[2]:
                    strength = min(1.0, abs(prev[2] - curr[2]) / 20)
                    result['bearish'] = True
                    result['strength'] = strength

        return result

    def _check_exit(
        self,
        pos: Position,
        high: float,
        low: float,
        close: float,
        bar: int,
        time: datetime,
        cfg: PhysicsConfig
    ) -> Optional[Tuple[float, str, float]]:
        """청산 체크"""
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
