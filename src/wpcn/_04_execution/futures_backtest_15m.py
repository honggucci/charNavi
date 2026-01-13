"""
선물 백테스트 엔진 v2.6.14 - 15분봉 지정가 주문 전용
======================================================

v3 대비 변경:
- 5분봉 X → 15분봉 전용
- 시장가 X → 지정가 진입 (Pending Order)
- SL/TP도 지정가 (체결 즉시 설정)
- 최대 보유 32봉 (8시간) - 펀딩비 회피

핵심 로직:
1. 시그널 발생 → PendingOrder 생성 (limit_price = 시그널 가격)
2. 다음 1~4봉 동안 bar_low <= limit_price <= bar_high 체크
3. 체결 시 SL/TP 즉시 설정 (ATR 기반 지정가)
4. 32봉 초과 → 시장가 청산 (time-stop)
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from wpcn._03_common._01_core.types import Theta, BacktestConfig
from wpcn.wyckoff.phases import detect_wyckoff_phase
from wpcn.wyckoff.box import box_engine_freeze


@dataclass
class FuturesConfig15m:
    """15분봉 지정가 백테스트 설정"""
    leverage: float = 3.0
    margin_mode: str = 'isolated'

    # 포지션 사이징
    position_pct: float = 0.03  # 자본의 3%

    # ATR 기반 SL/TP
    atr_period: int = 14
    atr_sl_mult: float = 1.5    # SL = entry ± ATR * 1.5
    atr_tp_mult: float = 2.5    # TP = entry ± ATR * 2.5

    # 지정가 주문 유효기간
    pending_order_max_bars: int = 4   # 4봉(1시간) 미체결 시 취소

    # 최대 보유 시간 (펀딩비 회피)
    max_hold_bars: int = 32           # 32봉 = 8시간

    # 비용
    maker_fee: float = 0.0002         # 지정가 수수료 0.02%
    taker_fee: float = 0.0005         # 시장가 수수료 0.05%
    slippage_pct: float = 0.0001      # 슬리피지 0.01%

    # 시그널 필터 (Wyckoff + 피보나치)
    fib_tolerance: float = 0.02       # 피보나치 레벨 허용 오차 2%
    entry_cooldown: int = 6           # 진입 후 쿨다운 (봉)
    rsi_period: int = 14              # RSI 기간 (미사용, 호환성)


@dataclass
class PendingOrder:
    """미체결 지정가 주문"""
    side: Literal['long', 'short']
    limit_price: float        # 지정가
    target_qty: float         # 목표 수량
    margin: float             # 증거금
    sl_price: float           # SL 지정가
    tp_price: float           # TP 지정가
    signal_bar: int           # 시그널 발생 봉
    signal_time: pd.Timestamp # 시그널 시간
    max_bars: int = 4         # 유효 봉 수
    atr_val: float = 0.0      # ATR 값 (디버깅용)

    def is_expired(self, current_bar: int) -> bool:
        """주문 만료 체크"""
        return (current_bar - self.signal_bar) > self.max_bars

    def check_fill(self, bar_low: float, bar_high: float) -> bool:
        """체결 조건 체크: 지정가가 봉의 범위 내에 있는지"""
        return bar_low <= self.limit_price <= bar_high


@dataclass
class Position15m:
    """15분봉 포지션"""
    side: Literal['long', 'short']
    entry_price: float
    qty: float
    margin: float
    leverage: float
    entry_bar: int
    entry_time: pd.Timestamp
    sl_price: float
    tp_price: float

    def check_sl_hit(self, bar_low: float, bar_high: float) -> bool:
        """SL 체결 체크"""
        if self.side == 'long':
            return bar_low <= self.sl_price
        else:  # short
            return bar_high >= self.sl_price

    def check_tp_hit(self, bar_low: float, bar_high: float) -> bool:
        """TP 체결 체크"""
        if self.side == 'long':
            return bar_high >= self.tp_price
        else:  # short
            return bar_low <= self.tp_price

    def check_time_stop(self, current_bar: int, max_hold: int) -> bool:
        """시간 청산 체크"""
        return (current_bar - self.entry_bar) >= max_hold

    def calc_pnl(self, exit_price: float) -> float:
        """PnL 계산 (qty는 롱 양수, 숏 음수)"""
        if self.side == 'long':
            return (exit_price - self.entry_price) * self.qty
        else:  # short (qty가 음수이므로 abs 사용)
            return (self.entry_price - exit_price) * abs(self.qty)


def calc_atr_15m(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """15분봉 ATR 계산 (shift(1) 적용 - lookahead 방지)"""
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    atr = tr.rolling(period).mean()
    return atr.shift(1)  # 완성된 봉의 ATR만 사용


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI 계산"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


# 피보나치 레벨 (Wyckoff Box 기반)
FIB_LEVELS_LONG = [0.236, 0.382, 0.500]   # 롱: 박스 하단 근처
FIB_LEVELS_SHORT = [0.618, 0.786]          # 숏: 박스 상단 근처

# Wyckoff direction 매핑
LONG_DIRECTIONS = ['accumulation', 're_accumulation', 'markup']
SHORT_DIRECTIONS = ['distribution', 're_distribution', 'markdown']


def simulate_futures_15m(
    df_15m: pd.DataFrame,
    theta: Theta,
    bt_cfg: BacktestConfig,
    futures_cfg: FuturesConfig15m,
    initial_equity: float = 10000.0,
    seed: int = 42,
    hmm_gate: Optional[Any] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    15분봉 지정가 주문 백테스트 엔진

    Args:
        df_15m: 15분봉 OHLCV 데이터
        theta: Wyckoff 파라미터
        bt_cfg: 백테스트 설정
        futures_cfg: 선물 설정
        initial_equity: 초기 자본
        seed: 랜덤 시드
        hmm_gate: HMM Entry Gate (선택)

    Returns:
        trades_df: 거래 내역
        equity_df: 자본 곡선
        stats: 통계
    """
    np.random.seed(seed)

    # === 지표 계산 ===
    phases_df = detect_wyckoff_phase(df_15m, theta)
    box_df = box_engine_freeze(df_15m, theta)
    atr_series = calc_atr_15m(df_15m, futures_cfg.atr_period)
    rsi_series = calc_rsi(df_15m['close'], futures_cfg.rsi_period)

    # === 상태 변수 ===
    cash = initial_equity
    position: Optional[Position15m] = None
    pending_order: Optional[PendingOrder] = None
    last_entry_bar: int = -999  # 마지막 진입 봉 (쿨다운용)

    trades: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    # === 통계 ===
    stats = {
        'signals_generated': 0,
        'orders_placed': 0,
        'orders_filled': 0,
        'orders_expired': 0,
        'sl_exits': 0,
        'tp_exits': 0,
        'time_stop_exits': 0,
        'hmm_gate_checks': 0,
        'hmm_gate_blocked': 0,
        'hmm_gate_blocked_transition': 0,
        'hmm_gate_blocked_short_permit': 0,
        'hmm_gate_blocked_long_permit': 0,
        'total_fees': 0.0,
    }

    idx = df_15m.index
    n = len(df_15m)

    # 워밍업 기간 (ATR, RSI 계산용)
    warmup = max(futures_cfg.atr_period + 1, 50)

    for i in range(warmup, n):
        t = idx[i]
        row = df_15m.iloc[i]
        o, h, l, c = row['open'], row['high'], row['low'], row['close']

        atr_val = atr_series.iloc[i] if not pd.isna(atr_series.iloc[i]) else 0.0
        rsi_val = rsi_series.iloc[i] if not pd.isna(rsi_series.iloc[i]) else 50.0

        # Wyckoff Phase
        phase = phases_df.iloc[i]['phase'] if i < len(phases_df) else 'unknown'

        # === 1. 포지션 청산 체크 (SL/TP/Time-stop) ===
        if position is not None:
            exit_price = None
            exit_type = None

            # SL 체크 (지정가)
            if position.check_sl_hit(l, h):
                exit_price = position.sl_price
                exit_type = 'SL'
                stats['sl_exits'] += 1

            # TP 체크 (지정가) - SL 미체결 시에만
            elif position.check_tp_hit(l, h):
                exit_price = position.tp_price
                exit_type = 'TP'
                stats['tp_exits'] += 1

            # Time-stop 체크 (시장가)
            elif position.check_time_stop(i, futures_cfg.max_hold_bars):
                exit_price = c  # 시장가 청산
                # 슬리피지 적용
                if position.side == 'long':
                    exit_price *= (1 - futures_cfg.slippage_pct)
                else:
                    exit_price *= (1 + futures_cfg.slippage_pct)
                exit_type = 'TIME_STOP'
                stats['time_stop_exits'] += 1

            if exit_price is not None:
                pnl = position.calc_pnl(exit_price)

                # 수수료 적용
                if exit_type == 'TIME_STOP':
                    fee = abs(position.qty * exit_price) * futures_cfg.taker_fee
                else:
                    fee = abs(position.qty * exit_price) * futures_cfg.maker_fee

                pnl -= fee
                stats['total_fees'] += fee

                pnl_pct = (pnl / position.margin) * 100 if position.margin > 0 else 0

                cash += position.margin + pnl

                trades.append({
                    'time': t,
                    'type': exit_type,
                    'side': position.side,
                    'entry_bar': position.entry_bar,
                    'exit_bar': i,
                    'hold_bars': i - position.entry_bar,
                    'entry_price': position.entry_price,
                    'exit_price': exit_price,
                    'sl_price': position.sl_price,
                    'tp_price': position.tp_price,
                    'qty': position.qty,
                    'margin': position.margin,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'fee': fee,
                    'leverage': position.leverage,
                })

                position = None

        # === 2. Pending Order 체결 체크 ===
        if pending_order is not None and position is None:
            # 만료 체크
            if pending_order.is_expired(i):
                stats['orders_expired'] += 1
                pending_order = None

            # 체결 체크
            elif pending_order.check_fill(l, h):
                # 체결!
                fill_price = pending_order.limit_price

                # Entry 수수료
                fee = abs(pending_order.target_qty * fill_price) * futures_cfg.maker_fee
                stats['total_fees'] += fee

                position = Position15m(
                    side=pending_order.side,
                    entry_price=fill_price,
                    qty=pending_order.target_qty,
                    margin=pending_order.margin,
                    leverage=futures_cfg.leverage,
                    entry_bar=i,
                    entry_time=t,
                    sl_price=pending_order.sl_price,
                    tp_price=pending_order.tp_price,
                )

                last_entry_bar = i  # 쿨다운 업데이트
                stats['orders_filled'] += 1
                pending_order = None

        # === 3. 시그널 감지 → Pending Order 생성 ===
        # 쿨다운 체크
        bars_since_last_entry = i - last_entry_bar if last_entry_bar >= 0 else 999

        if position is None and pending_order is None and bars_since_last_entry >= futures_cfg.entry_cooldown:
            signal_side = None
            limit_price = None

            # Wyckoff direction + 피보나치 레벨 기반 시그널
            # direction은 phases_df에서 가져옴
            direction = phases_df.iloc[i]['direction'] if i < len(phases_df) and 'direction' in phases_df.columns else 'unknown'

            # Box 정보
            box_low = box_df.iloc[i]['box_low'] if i < len(box_df) and 'box_low' in box_df.columns else np.nan
            box_high = box_df.iloc[i]['box_high'] if i < len(box_df) and 'box_high' in box_df.columns else np.nan

            if not pd.isna(box_low) and not pd.isna(box_high) and box_high > box_low:
                box_width = box_high - box_low
                box_pos = (c - box_low) / box_width if box_width > 0 else 0.5

                # Long: accumulation/markup direction + 박스 하단 (fib 0.236~0.5)
                if direction in LONG_DIRECTIONS and box_pos < 0.55:
                    for fib_pct in FIB_LEVELS_LONG:
                        fib_price = box_low + box_width * fib_pct
                        if abs(c - fib_price) / c < futures_cfg.fib_tolerance:
                            signal_side = 'long'
                            limit_price = fib_price  # 피보나치 레벨에 지정가
                            break

                # Short: distribution/markdown direction + 박스 상단 (fib 0.618~0.786)
                elif direction in SHORT_DIRECTIONS and box_pos > 0.45:
                    for fib_pct in FIB_LEVELS_SHORT:
                        fib_price = box_low + box_width * fib_pct
                        if abs(c - fib_price) / c < futures_cfg.fib_tolerance:
                            signal_side = 'short'
                            limit_price = fib_price  # 피보나치 레벨에 지정가
                            break

            if signal_side is not None:
                stats['signals_generated'] += 1

                # HMM Entry Gate 체크
                if hmm_gate is not None:
                    stats['hmm_gate_checks'] += 1
                    gate_result = hmm_gate.check_entry(t, signal_side)

                    if not gate_result.allowed:
                        stats['hmm_gate_blocked'] += 1
                        reason = gate_result.blocked_reason or ''
                        if 'transition' in reason:
                            stats['hmm_gate_blocked_transition'] += 1
                        elif 'short' in reason:
                            stats['hmm_gate_blocked_short_permit'] += 1
                        elif 'long' in reason:
                            stats['hmm_gate_blocked_long_permit'] += 1
                        signal_side = None  # 시그널 취소

                if signal_side is not None and atr_val > 0 and limit_price is not None:
                    # 포지션 사이징
                    margin = cash * futures_cfg.position_pct
                    notional = margin * futures_cfg.leverage
                    qty = notional / limit_price

                    # SL/TP 계산 (ATR 기반, 지정가 기준)
                    if signal_side == 'long':
                        sl_price = limit_price - atr_val * futures_cfg.atr_sl_mult
                        tp_price = limit_price + atr_val * futures_cfg.atr_tp_mult
                    else:  # short
                        sl_price = limit_price + atr_val * futures_cfg.atr_sl_mult
                        tp_price = limit_price - atr_val * futures_cfg.atr_tp_mult

                    pending_order = PendingOrder(
                        side=signal_side,
                        limit_price=limit_price,
                        target_qty=qty if signal_side == 'long' else -qty,
                        margin=margin,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        signal_bar=i,
                        signal_time=t,
                        max_bars=futures_cfg.pending_order_max_bars,
                        atr_val=atr_val,
                    )

                    stats['orders_placed'] += 1

        # === 4. Equity 기록 ===
        unrealized_pnl = 0.0
        if position is not None:
            unrealized_pnl = position.calc_pnl(c)

        equity = cash + (position.margin if position else 0) + unrealized_pnl

        equity_rows.append({
            'time': t,
            'bar': i,
            'cash': cash,
            'position_margin': position.margin if position else 0,
            'unrealized_pnl': unrealized_pnl,
            'equity': equity,
            'has_position': position is not None,
            'has_pending': pending_order is not None,
        })

    # === 결과 DataFrame ===
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(equity_rows)

    # === 최종 통계 계산 ===
    if len(trades_df) > 0:
        stats['total_trades'] = len(trades_df)
        stats['win_trades'] = len(trades_df[trades_df['pnl'] > 0])
        stats['lose_trades'] = len(trades_df[trades_df['pnl'] <= 0])
        stats['win_rate'] = stats['win_trades'] / stats['total_trades'] * 100

        stats['total_pnl'] = trades_df['pnl'].sum()
        stats['avg_pnl'] = trades_df['pnl'].mean()
        stats['avg_pnl_pct'] = trades_df['pnl_pct'].mean()

        stats['avg_hold_bars'] = trades_df['hold_bars'].mean()

        # SL/TP/Time-stop 비율
        stats['sl_rate'] = stats['sl_exits'] / stats['total_trades'] * 100
        stats['tp_rate'] = stats['tp_exits'] / stats['total_trades'] * 100
        stats['time_stop_rate'] = stats['time_stop_exits'] / stats['total_trades'] * 100

        # 주문 체결률
        if stats['orders_placed'] > 0:
            stats['fill_rate'] = stats['orders_filled'] / stats['orders_placed'] * 100
            stats['expire_rate'] = stats['orders_expired'] / stats['orders_placed'] * 100
        else:
            stats['fill_rate'] = 0.0
            stats['expire_rate'] = 0.0
    else:
        stats['total_trades'] = 0
        stats['win_rate'] = 0.0
        stats['total_pnl'] = 0.0
        stats['fill_rate'] = 0.0

    # Max Drawdown
    if len(equity_df) > 0:
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        stats['max_dd'] = equity_df['drawdown'].min() * 100
        stats['final_equity'] = equity_df['equity'].iloc[-1]
    else:
        stats['max_dd'] = 0.0
        stats['final_equity'] = initial_equity

    # HMM Gate 통계
    if hmm_gate is not None:
        stats['hmm_block_rate'] = (
            stats['hmm_gate_blocked'] / stats['hmm_gate_checks'] * 100
            if stats['hmm_gate_checks'] > 0 else 0.0
        )

    return trades_df, equity_df, stats


def run_15m_backtest_from_5m(
    df_5m: pd.DataFrame,
    theta: Theta,
    bt_cfg: BacktestConfig,
    futures_cfg: Optional[FuturesConfig15m] = None,
    initial_equity: float = 10000.0,
    seed: int = 42,
    hmm_gate: Optional[Any] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    5분봉 데이터에서 15분봉 리샘플링 후 백테스트 실행

    기존 인프라(5분봉 데이터 로더)와 호환성 유지용 헬퍼 함수
    """
    # 5분봉 → 15분봉 리샘플링
    df_15m = df_5m.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    if futures_cfg is None:
        futures_cfg = FuturesConfig15m()

    return simulate_futures_15m(
        df_15m=df_15m,
        theta=theta,
        bt_cfg=bt_cfg,
        futures_cfg=futures_cfg,
        initial_equity=initial_equity,
        seed=seed,
        hmm_gate=hmm_gate,
    )