"""
선물 백테스트 엔진 V5 - 단순화된 추세 추종 전략
Futures Backtest Engine V5 - Simplified Trend Following

핵심 전략:
- 4시간봉 EMA 크로스로 추세 확인
- 1시간봉 구조 돌파로 진입 타이밍
- ATR 기반 동적 TP/SL
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class TrendConfig:
    """추세 추종 백테스트 설정"""
    leverage: float = 3.0
    position_pct: float = 0.10  # 10%

    # ATR 기반 TP/SL
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 4.0

    # EMA 설정
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200  # 대추세

    # 시간 설정 (5분봉 기준)
    warmup_bars: int = 500
    cooldown_bars: int = 36  # 3시간
    max_hold_bars: int = 288  # 24시간

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


class TrendBacktestEngine:
    """
    단순화된 추세 추종 백테스트

    규칙:
    1. 4H EMA(20) > EMA(50) > EMA(200) = Long만 진입
    2. 4H EMA(20) < EMA(50) < EMA(200) = Short만 진입
    3. 1H에서 구조 돌파 시 진입
    4. ATR 기반 TP/SL
    """

    def __init__(self, config: TrendConfig = None):
        self.config = config or TrendConfig()
        self.position: Optional[Position] = None
        self.trades: List[Dict] = []
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
        self.equity_history = []

        cash = initial_capital
        last_entry_bar = -cfg.cooldown_bars - 1

        # 리샘플링
        df_1h = self._resample(df_5m, '1h')
        df_4h = self._resample(df_5m, '4h')

        # EMA 계산 (4시간봉)
        df_4h['ema_fast'] = df_4h['close'].ewm(span=cfg.ema_fast).mean()
        df_4h['ema_slow'] = df_4h['close'].ewm(span=cfg.ema_slow).mean()
        df_4h['ema_trend'] = df_4h['close'].ewm(span=cfg.ema_trend).mean()

        # 1시간봉 구조
        df_1h['swing_high'] = df_1h['high'].rolling(20).max()
        df_1h['swing_low'] = df_1h['low'].rolling(20).min()
        df_1h['breakout_up'] = df_1h['close'] > df_1h['swing_high'].shift(1)
        df_1h['breakout_down'] = df_1h['close'] < df_1h['swing_low'].shift(1)

        # ATR 계산 (5분봉)
        df_5m['tr'] = np.maximum(
            df_5m['high'] - df_5m['low'],
            np.maximum(
                abs(df_5m['high'] - df_5m['close'].shift(1)),
                abs(df_5m['low'] - df_5m['close'].shift(1))
            )
        )
        df_5m['atr'] = df_5m['tr'].rolling(14).mean()

        n = len(df_5m)
        idx = df_5m.index

        print(f"Running Trend backtest: {n} bars, {symbol}")

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
                        'tp_price': self.position.tp_price,
                        'sl_price': self.position.sl_price,
                        'pnl': pnl,
                        'pnl_pct': pnl / self.position.margin * 100,
                        'entry_time': self.position.entry_time,
                        'exit_time': t,
                        'exit_reason': exit_reason,
                        'hold_bars': i - self.position.entry_bar
                    })
                    self.position = None

            # === 2. 신호 체크 ===
            if self.position is None and i - last_entry_bar >= cfg.cooldown_bars and cash > 0:
                signal = self._check_signal(df_4h, df_1h, t)

                if signal:
                    entry_price = c * (1 + cfg.slippage if signal == 'long' else 1 - cfg.slippage)

                    if signal == 'long':
                        sl_price = entry_price - current_atr * cfg.sl_atr_mult
                        tp_price = entry_price + current_atr * cfg.tp_atr_mult
                    else:
                        sl_price = entry_price + current_atr * cfg.sl_atr_mult
                        tp_price = entry_price - current_atr * cfg.tp_atr_mult

                    position_value = cash * cfg.position_pct * cfg.leverage
                    quantity = position_value / entry_price
                    margin = position_value / cfg.leverage
                    fee = position_value * cfg.maker_fee

                    self.position = Position(
                        side=signal,
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

    def _check_signal(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        timestamp: datetime
    ) -> Optional[str]:
        """신호 체크 (완료된 봉만 사용)"""
        # 중요: 완료된 이전 봉만 참조해야 look-ahead bias 방지
        # 현재 시간의 봉이 아닌, 완료된 이전 봉을 사용
        from datetime import timedelta

        # 이전 완료된 4시간봉 (현재 봉 - 4시간)
        t_4h_prev = timestamp.floor('4h') - timedelta(hours=4)
        # 이전 완료된 1시간봉 (현재 봉 - 1시간)
        t_1h_prev = timestamp.floor('1h') - timedelta(hours=1)

        # 4시간봉 추세 확인 (완료된 이전 봉)
        if t_4h_prev not in df_4h.index:
            return None

        row_4h = df_4h.loc[t_4h_prev]
        ema_fast = row_4h.get('ema_fast')
        ema_slow = row_4h.get('ema_slow')
        ema_trend = row_4h.get('ema_trend')

        if pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(ema_trend):
            return None

        # 추세 방향
        bullish_trend = ema_fast > ema_slow > ema_trend
        bearish_trend = ema_fast < ema_slow < ema_trend

        if not bullish_trend and not bearish_trend:
            return None

        # 1시간봉 돌파 확인 (완료된 이전 봉)
        if t_1h_prev not in df_1h.index:
            return None

        row_1h = df_1h.loc[t_1h_prev]
        breakout_up = row_1h.get('breakout_up', False)
        breakout_down = row_1h.get('breakout_down', False)

        # 신호 발생
        if bullish_trend and breakout_up:
            return 'long'
        elif bearish_trend and breakout_down:
            return 'short'

        return None

    def _check_exit(
        self,
        pos: Position,
        high: float,
        low: float,
        close: float,
        bar: int,
        time: datetime,
        cfg: TrendConfig
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
            'avg_pnl': avg_pnl
        }

        if len(trades_df) > 0 and 'exit_reason' in trades_df.columns:
            stats['tp_exits'] = len(trades_df[trades_df['exit_reason'] == 'tp'])
            stats['sl_exits'] = len(trades_df[trades_df['exit_reason'] == 'sl'])
            stats['timeout_exits'] = len(trades_df[trades_df['exit_reason'] == 'timeout'])
            stats['liquidations'] = len(trades_df[trades_df['exit_reason'] == 'liquidation'])

        return {
            'equity_df': equity_df,
            'trades_df': trades_df,
            'stats': stats
        }
