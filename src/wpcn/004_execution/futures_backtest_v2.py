"""
선물 백테스트 엔진 V2
Futures Backtest Engine V2

개선사항:
- 15분봉 축적 매매 (여러 번 진입 가능)
- 5분봉 다이버전스 조건 완화
- 6시간 시간 청산
- 청산 로직, 펀딩비 포함
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from wpcn.core.types import Theta, BacktestCosts, BacktestConfig
from wpcn.execution.cost import apply_costs
from wpcn.features.indicators import atr, rsi, detect_rsi_divergence, stoch_rsi
from wpcn.wyckoff.phases import detect_wyckoff_phase, DIRECTION_RATIOS, FIB_LEVELS
from wpcn.wyckoff.box import box_engine_freeze


@dataclass
class FuturesConfigV2:
    """선물 백테스트 설정 V2"""
    leverage: float = 10.0
    margin_mode: str = 'isolated'
    maintenance_margin_rate: float = 0.005  # 0.5%
    funding_rate: float = 0.0001  # 0.01% per 8 hours
    funding_interval_bars: int = 96  # 5분봉 기준 8시간 = 96봉

    # 포지션 사이징 - 15분봉 축적
    accumulation_pct_15m: float = 0.03  # 3%씩 축적
    max_accumulation_pct: float = 0.15  # 최대 15% 축적

    # 포지션 사이징 - 5분봉
    scalping_pct_5m: float = 0.02  # 2%씩 진입

    # TP/SL - 15분봉 (레버리지 적용 전 기준)
    tp_pct_15m: float = 0.015  # +1.5%
    sl_pct_15m: float = 0.010  # -1.0%

    # TP/SL - 5분봉
    tp_pct_5m: float = 0.008  # +0.8%
    sl_pct_5m: float = 0.005  # -0.5%

    # 시간 청산 (5분봉 기준)
    max_hold_bars_15m: int = 72  # 15분봉 포지션: 6시간 = 72봉 (5분봉)
    max_hold_bars_5m: int = 36   # 5분봉 포지션: 3시간 = 36봉

    # 5분봉 다이버전스 필터 (완화된 조건)
    rsi_oversold: float = 45  # RSI < 45 (기존 40)
    rsi_overbought: float = 55  # RSI > 55 (기존 60)
    stoch_oversold: float = 40  # Stoch < 40 (기존 30)
    stoch_overbought: float = 60  # Stoch > 60 (기존 70)
    divergence_lookback: int = 20

    # 축적 쿨다운
    accumulation_cooldown: int = 6  # 5분봉 6봉 = 30분 쿨다운


@dataclass
class AccumulatedPosition:
    """축적 포지션 (여러 진입을 합산)"""
    side: str  # 'long' or 'short'
    total_margin: float = 0.0
    total_size: float = 0.0
    avg_entry_price: float = 0.0
    first_entry_bar: int = 0
    last_entry_bar: int = 0
    leverage: float = 10.0
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def add_entry(self, margin: float, size: float, price: float, bar: int, event: str):
        """포지션 추가"""
        if self.total_size == 0:
            self.avg_entry_price = price
            self.first_entry_bar = bar
        else:
            # 평균 진입가 계산
            total_notional = self.avg_entry_price * abs(self.total_size) + price * abs(size)
            new_total_size = abs(self.total_size) + abs(size)
            self.avg_entry_price = total_notional / new_total_size

        self.total_margin += margin
        self.total_size += size
        self.last_entry_bar = bar
        self.entries.append({
            'bar': bar, 'price': price, 'margin': margin, 'size': size, 'event': event
        })

    @property
    def liquidation_price(self) -> float:
        """청산가 계산"""
        if self.side == 'long':
            return self.avg_entry_price * (1 - 1/self.leverage + 0.005)
        else:
            return self.avg_entry_price * (1 + 1/self.leverage - 0.005)

    @property
    def tp_price(self) -> float:
        """익절가"""
        if self.side == 'long':
            return self.avg_entry_price * 1.015
        else:
            return self.avg_entry_price * 0.985

    @property
    def sl_price(self) -> float:
        """손절가"""
        if self.side == 'long':
            return self.avg_entry_price * 0.990
        else:
            return self.avg_entry_price * 1.010


def resample_5m_to_15m(df_5m: pd.DataFrame) -> pd.DataFrame:
    """5분봉을 15분봉으로 리샘플링"""
    df_15m = df_5m.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return df_15m


def detect_5m_divergence(df_5m: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """5분봉 RSI 다이버전스 감지"""
    div_df = detect_rsi_divergence(df_5m, rsi_len=14, lookback=lookback)
    return pd.DataFrame(div_df, index=df_5m.index)


def simulate_futures_v2(
    df_5m: pd.DataFrame,
    theta: Theta,
    costs: BacktestCosts,
    bt_cfg: BacktestConfig,
    futures_cfg: FuturesConfigV2,
    initial_equity: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    선물 백테스트 V2 - 축적 매매 지원

    개선사항:
    - 15분봉 축적 매매 (여러 번 진입, 평균단가 계산)
    - 5분봉 다이버전스 조건 완화
    - 6시간 시간 청산
    """
    # 5분봉 -> 15분봉 리샘플링
    df_15m = resample_5m_to_15m(df_5m)

    # 15분봉 Wyckoff 페이즈 감지
    phases_df = detect_wyckoff_phase(df_15m, theta)

    # 15분봉 박스 감지
    box_df = box_engine_freeze(df_15m, theta)

    # 5분봉 다이버전스 감지
    div_5m = detect_5m_divergence(df_5m, futures_cfg.divergence_lookback)

    # 5분봉 RSI
    rsi_5m = rsi(df_5m['close'], 14)

    # 5분봉 Stochastic RSI
    stoch_5m = stoch_rsi(df_5m['close'], 14, 14, 3, 3)

    # 상태 변수
    cash = initial_equity

    # === 15분봉 축적 포지션 ===
    accum_long: Optional[AccumulatedPosition] = None
    accum_short: Optional[AccumulatedPosition] = None
    last_accum_bar = -100

    # === 5분봉 포지션 ===
    position_5m_long: Optional[Dict[str, Any]] = None
    position_5m_short: Optional[Dict[str, Any]] = None
    last_5m_entry_bar = -100

    trades: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    # 펀딩비 카운터
    bars_since_funding = 0
    total_funding_paid = 0.0

    # 청산 횟수
    liquidation_count_15m = 0
    liquidation_count_5m = 0

    # 축적 횟수
    accumulation_count = 0

    idx_5m = df_5m.index
    n = len(df_5m)

    for i in range(50, n):
        t = idx_5m[i]
        o, h, l, c = df_5m.iloc[i][['open', 'high', 'low', 'close']]

        # 현재 5분봉에 해당하는 15분봉 찾기
        t_15m = t.floor('15min')

        # 15분봉 정보 가져오기
        if t_15m in phases_df.index:
            phase_row = phases_df.loc[t_15m]
            phase = phase_row['phase']
            direction = phase_row['direction']

            if t_15m in box_df.index:
                box_row = box_df.loc[t_15m]
                box_low = box_row['box_low']
                box_high = box_row['box_high']
                box_width = box_high - box_low
            else:
                box_low = box_high = box_width = np.nan
        else:
            phase = 'unknown'
            direction = 'unknown'
            box_low = box_high = box_width = np.nan

        # 5분봉 다이버전스/RSI/Stoch 정보
        bullish_div = div_5m.iloc[i]['bullish_div'] if i < len(div_5m) and 'bullish_div' in div_5m.columns else False
        bearish_div = div_5m.iloc[i]['bearish_div'] if i < len(div_5m) and 'bearish_div' in div_5m.columns else False
        curr_rsi = rsi_5m.iloc[i] if i < len(rsi_5m) else 50
        curr_stoch_k = stoch_5m['stoch_k'].iloc[i] if i < len(stoch_5m) else 50

        # === 펀딩비 처리 ===
        bars_since_funding += 1
        if bars_since_funding >= futures_cfg.funding_interval_bars:
            # 축적 롱 포지션 펀딩비
            if accum_long is not None and accum_long.total_size > 0:
                position_value = abs(accum_long.total_size) * c
                funding_fee = position_value * futures_cfg.funding_rate
                cash -= funding_fee
                total_funding_paid += funding_fee

            # 축적 숏 포지션 펀딩비
            if accum_short is not None and accum_short.total_size < 0:
                position_value = abs(accum_short.total_size) * c
                funding_fee = -position_value * futures_cfg.funding_rate  # 숏은 받음
                cash -= funding_fee
                total_funding_paid += funding_fee

            # 5분봉 포지션 펀딩비
            if position_5m_long is not None:
                position_value = position_5m_long['size'] * c
                funding_fee = position_value * futures_cfg.funding_rate
                cash -= funding_fee
                total_funding_paid += funding_fee

            if position_5m_short is not None:
                position_value = abs(position_5m_short['size']) * c
                funding_fee = -position_value * futures_cfg.funding_rate
                cash -= funding_fee
                total_funding_paid += funding_fee

            bars_since_funding = 0

        # ========================================
        # === 15분봉 축적 롱 포지션 관리 ===
        # ========================================
        if accum_long is not None and accum_long.total_size > 0:
            closed = False

            # 1. 청산 체크
            if l <= accum_long.liquidation_price:
                trades.append({
                    'time': t, 'type': 'LIQUIDATION', 'side': 'long', 'timeframe': '15m_accum',
                    'entry_price': accum_long.avg_entry_price,
                    'exit_price': accum_long.liquidation_price,
                    'pnl': -accum_long.total_margin, 'pnl_pct': -100.0,
                    'leverage': accum_long.leverage, 'num_entries': len(accum_long.entries)
                })
                cash += 0  # 마진 전액 손실
                accum_long = None
                liquidation_count_15m += 1
                closed = True

            # 2. 손절 체크
            if not closed and accum_long is not None and l <= accum_long.sl_price:
                exit_px = apply_costs(accum_long.sl_price, costs)
                pnl = (exit_px - accum_long.avg_entry_price) * accum_long.total_size
                pnl_pct = pnl / accum_long.total_margin * 100
                cash += accum_long.total_margin + pnl
                trades.append({
                    'time': t, 'type': 'SL', 'side': 'long', 'timeframe': '15m_accum',
                    'entry_price': accum_long.avg_entry_price, 'exit_price': exit_px,
                    'pnl': pnl, 'pnl_pct': pnl_pct,
                    'leverage': accum_long.leverage, 'num_entries': len(accum_long.entries)
                })
                accum_long = None
                closed = True

            # 3. 익절 체크
            if not closed and accum_long is not None and h >= accum_long.tp_price:
                exit_px = apply_costs(accum_long.tp_price, costs)
                pnl = (exit_px - accum_long.avg_entry_price) * accum_long.total_size
                pnl_pct = pnl / accum_long.total_margin * 100
                cash += accum_long.total_margin + pnl
                trades.append({
                    'time': t, 'type': 'TP', 'side': 'long', 'timeframe': '15m_accum',
                    'entry_price': accum_long.avg_entry_price, 'exit_price': exit_px,
                    'pnl': pnl, 'pnl_pct': pnl_pct,
                    'leverage': accum_long.leverage, 'num_entries': len(accum_long.entries)
                })
                accum_long = None
                closed = True

            # 4. 시간 청산 (6시간 = 72봉)
            if not closed and accum_long is not None and (i - accum_long.first_entry_bar) >= futures_cfg.max_hold_bars_15m:
                exit_px = apply_costs(c, costs)
                pnl = (exit_px - accum_long.avg_entry_price) * accum_long.total_size
                pnl_pct = pnl / accum_long.total_margin * 100
                cash += accum_long.total_margin + pnl
                trades.append({
                    'time': t, 'type': 'TIME_EXIT_6H', 'side': 'long', 'timeframe': '15m_accum',
                    'entry_price': accum_long.avg_entry_price, 'exit_price': exit_px,
                    'pnl': pnl, 'pnl_pct': pnl_pct,
                    'leverage': accum_long.leverage, 'num_entries': len(accum_long.entries)
                })
                accum_long = None

        # ========================================
        # === 15분봉 축적 숏 포지션 관리 ===
        # ========================================
        if accum_short is not None and accum_short.total_size < 0:
            closed = False

            # 1. 청산 체크
            if h >= accum_short.liquidation_price:
                trades.append({
                    'time': t, 'type': 'LIQUIDATION', 'side': 'short', 'timeframe': '15m_accum',
                    'entry_price': accum_short.avg_entry_price,
                    'exit_price': accum_short.liquidation_price,
                    'pnl': -accum_short.total_margin, 'pnl_pct': -100.0,
                    'leverage': accum_short.leverage, 'num_entries': len(accum_short.entries)
                })
                cash += 0
                accum_short = None
                liquidation_count_15m += 1
                closed = True

            # 2. 손절 체크
            if not closed and accum_short is not None and h >= accum_short.sl_price:
                exit_px = apply_costs(accum_short.sl_price, costs)
                pnl = (accum_short.avg_entry_price - exit_px) * abs(accum_short.total_size)
                pnl_pct = pnl / accum_short.total_margin * 100
                cash += accum_short.total_margin + pnl
                trades.append({
                    'time': t, 'type': 'SL', 'side': 'short', 'timeframe': '15m_accum',
                    'entry_price': accum_short.avg_entry_price, 'exit_price': exit_px,
                    'pnl': pnl, 'pnl_pct': pnl_pct,
                    'leverage': accum_short.leverage, 'num_entries': len(accum_short.entries)
                })
                accum_short = None
                closed = True

            # 3. 익절 체크
            if not closed and accum_short is not None and l <= accum_short.tp_price:
                exit_px = apply_costs(accum_short.tp_price, costs)
                pnl = (accum_short.avg_entry_price - exit_px) * abs(accum_short.total_size)
                pnl_pct = pnl / accum_short.total_margin * 100
                cash += accum_short.total_margin + pnl
                trades.append({
                    'time': t, 'type': 'TP', 'side': 'short', 'timeframe': '15m_accum',
                    'entry_price': accum_short.avg_entry_price, 'exit_price': exit_px,
                    'pnl': pnl, 'pnl_pct': pnl_pct,
                    'leverage': accum_short.leverage, 'num_entries': len(accum_short.entries)
                })
                accum_short = None
                closed = True

            # 4. 시간 청산 (6시간)
            if not closed and accum_short is not None and (i - accum_short.first_entry_bar) >= futures_cfg.max_hold_bars_15m:
                exit_px = apply_costs(c, costs)
                pnl = (accum_short.avg_entry_price - exit_px) * abs(accum_short.total_size)
                pnl_pct = pnl / accum_short.total_margin * 100
                cash += accum_short.total_margin + pnl
                trades.append({
                    'time': t, 'type': 'TIME_EXIT_6H', 'side': 'short', 'timeframe': '15m_accum',
                    'entry_price': accum_short.avg_entry_price, 'exit_price': exit_px,
                    'pnl': pnl, 'pnl_pct': pnl_pct,
                    'leverage': accum_short.leverage, 'num_entries': len(accum_short.entries)
                })
                accum_short = None

        # ========================================
        # === 5분봉 롱 포지션 관리 ===
        # ========================================
        if position_5m_long is not None:
            closed = False
            pos = position_5m_long

            # 청산 체크
            liq_price = pos['entry_price'] * (1 - 1/futures_cfg.leverage + 0.005)
            if l <= liq_price:
                trades.append({
                    'time': t, 'type': 'LIQUIDATION', 'side': 'long', 'timeframe': '5m',
                    'entry_price': pos['entry_price'], 'exit_price': liq_price,
                    'pnl': -pos['margin'], 'pnl_pct': -100.0,
                    'leverage': futures_cfg.leverage
                })
                position_5m_long = None
                liquidation_count_5m += 1
                closed = True

            # 손절
            sl_price = pos['entry_price'] * (1 - futures_cfg.sl_pct_5m)
            if not closed and l <= sl_price:
                exit_px = apply_costs(sl_price, costs)
                pnl = (exit_px - pos['entry_price']) * pos['size']
                pnl_pct = pnl / pos['margin'] * 100
                cash += pos['margin'] + pnl
                trades.append({
                    'time': t, 'type': 'SL', 'side': 'long', 'timeframe': '5m',
                    'entry_price': pos['entry_price'], 'exit_price': exit_px,
                    'pnl': pnl, 'pnl_pct': pnl_pct, 'leverage': futures_cfg.leverage
                })
                position_5m_long = None
                closed = True

            # 익절
            tp_price = pos['entry_price'] * (1 + futures_cfg.tp_pct_5m)
            if not closed and h >= tp_price:
                exit_px = apply_costs(tp_price, costs)
                pnl = (exit_px - pos['entry_price']) * pos['size']
                pnl_pct = pnl / pos['margin'] * 100
                cash += pos['margin'] + pnl
                trades.append({
                    'time': t, 'type': 'TP', 'side': 'long', 'timeframe': '5m',
                    'entry_price': pos['entry_price'], 'exit_price': exit_px,
                    'pnl': pnl, 'pnl_pct': pnl_pct, 'leverage': futures_cfg.leverage
                })
                position_5m_long = None
                closed = True

            # 시간 청산 (3시간)
            if not closed and position_5m_long is not None and (i - pos['entry_bar']) >= futures_cfg.max_hold_bars_5m:
                exit_px = apply_costs(c, costs)
                pnl = (exit_px - pos['entry_price']) * pos['size']
                pnl_pct = pnl / pos['margin'] * 100
                cash += pos['margin'] + pnl
                trades.append({
                    'time': t, 'type': 'TIME_EXIT_3H', 'side': 'long', 'timeframe': '5m',
                    'entry_price': pos['entry_price'], 'exit_price': exit_px,
                    'pnl': pnl, 'pnl_pct': pnl_pct, 'leverage': futures_cfg.leverage
                })
                position_5m_long = None

        # ========================================
        # === 5분봉 숏 포지션 관리 ===
        # ========================================
        if position_5m_short is not None:
            closed = False
            pos = position_5m_short

            # 청산 체크
            liq_price = pos['entry_price'] * (1 + 1/futures_cfg.leverage - 0.005)
            if h >= liq_price:
                trades.append({
                    'time': t, 'type': 'LIQUIDATION', 'side': 'short', 'timeframe': '5m',
                    'entry_price': pos['entry_price'], 'exit_price': liq_price,
                    'pnl': -pos['margin'], 'pnl_pct': -100.0,
                    'leverage': futures_cfg.leverage
                })
                position_5m_short = None
                liquidation_count_5m += 1
                closed = True

            # 손절
            sl_price = pos['entry_price'] * (1 + futures_cfg.sl_pct_5m)
            if not closed and h >= sl_price:
                exit_px = apply_costs(sl_price, costs)
                pnl = (pos['entry_price'] - exit_px) * abs(pos['size'])
                pnl_pct = pnl / pos['margin'] * 100
                cash += pos['margin'] + pnl
                trades.append({
                    'time': t, 'type': 'SL', 'side': 'short', 'timeframe': '5m',
                    'entry_price': pos['entry_price'], 'exit_price': exit_px,
                    'pnl': pnl, 'pnl_pct': pnl_pct, 'leverage': futures_cfg.leverage
                })
                position_5m_short = None
                closed = True

            # 익절
            tp_price = pos['entry_price'] * (1 - futures_cfg.tp_pct_5m)
            if not closed and l <= tp_price:
                exit_px = apply_costs(tp_price, costs)
                pnl = (pos['entry_price'] - exit_px) * abs(pos['size'])
                pnl_pct = pnl / pos['margin'] * 100
                cash += pos['margin'] + pnl
                trades.append({
                    'time': t, 'type': 'TP', 'side': 'short', 'timeframe': '5m',
                    'entry_price': pos['entry_price'], 'exit_price': exit_px,
                    'pnl': pnl, 'pnl_pct': pnl_pct, 'leverage': futures_cfg.leverage
                })
                position_5m_short = None
                closed = True

            # 시간 청산 (3시간)
            if not closed and position_5m_short is not None and (i - pos['entry_bar']) >= futures_cfg.max_hold_bars_5m:
                exit_px = apply_costs(c, costs)
                pnl = (pos['entry_price'] - exit_px) * abs(pos['size'])
                pnl_pct = pnl / pos['margin'] * 100
                cash += pos['margin'] + pnl
                trades.append({
                    'time': t, 'type': 'TIME_EXIT_3H', 'side': 'short', 'timeframe': '5m',
                    'entry_price': pos['entry_price'], 'exit_price': exit_px,
                    'pnl': pnl, 'pnl_pct': pnl_pct, 'leverage': futures_cfg.leverage
                })
                position_5m_short = None

        # ========================================
        # === 15분봉 축적 진입 (3봉마다 = 15분봉 마감) ===
        # ========================================
        is_15m_close = (i % 3 == 2)

        if is_15m_close and (i - last_accum_bar) >= futures_cfg.accumulation_cooldown:
            if not np.isnan(box_low) and box_width > 0:
                ratios = DIRECTION_RATIOS.get(direction, DIRECTION_RATIOS['unknown'])
                box_pos = (c - box_low) / box_width

                # 현재 총 축적 마진 계산
                current_long_margin = accum_long.total_margin if accum_long else 0
                current_short_margin = accum_short.total_margin if accum_short else 0

                # === 롱 축적: Accumulation 페이즈 ===
                if direction in ['accumulation', 're_accumulation'] and box_pos < 0.5:
                    # 축적 한도 체크
                    if current_long_margin < initial_equity * futures_cfg.max_accumulation_pct:
                        for fib_name, fib_pct in [('fib_236', 0.236), ('fib_382', 0.382), ('fib_500', 0.500)]:
                            fib_price = box_low + box_width * fib_pct
                            if abs(c - fib_price) / c < 0.01:  # 1% 이내 (완화)
                                position_pct = futures_cfg.accumulation_pct_15m * ratios['long']
                                if position_pct > 0.001 and cash > 0.01:
                                    margin = cash * position_pct
                                    fill_px = apply_costs(c, costs)
                                    notional = margin * futures_cfg.leverage
                                    size = notional / fill_px

                                    if accum_long is None:
                                        accum_long = AccumulatedPosition(
                                            side='long', leverage=futures_cfg.leverage
                                        )
                                    accum_long.add_entry(margin, size, fill_px, i, f'accum_{fib_name}_long')
                                    cash -= margin
                                    accumulation_count += 1

                                    trades.append({
                                        'time': t, 'type': 'ACCUM_ENTRY', 'side': 'long',
                                        'timeframe': '15m_accum', 'entry_price': fill_px,
                                        'margin': margin, 'total_margin': accum_long.total_margin,
                                        'num_entries': len(accum_long.entries)
                                    })
                                    last_accum_bar = i
                                break

                # === 숏 축적: Distribution 페이즈 ===
                if direction in ['distribution', 're_distribution'] and box_pos > 0.5:
                    if current_short_margin < initial_equity * futures_cfg.max_accumulation_pct:
                        for fib_name, fib_pct in [('fib_618', 0.618), ('fib_786', 0.786)]:
                            fib_price = box_low + box_width * fib_pct
                            if abs(c - fib_price) / c < 0.01:
                                position_pct = futures_cfg.accumulation_pct_15m * ratios['short']
                                if position_pct > 0.001 and cash > 0.01:
                                    margin = cash * position_pct
                                    fill_px = apply_costs(c, costs)
                                    notional = margin * futures_cfg.leverage
                                    size = -notional / fill_px

                                    if accum_short is None:
                                        accum_short = AccumulatedPosition(
                                            side='short', leverage=futures_cfg.leverage
                                        )
                                    accum_short.add_entry(margin, size, fill_px, i, f'accum_{fib_name}_short')
                                    cash -= margin
                                    accumulation_count += 1

                                    trades.append({
                                        'time': t, 'type': 'ACCUM_ENTRY', 'side': 'short',
                                        'timeframe': '15m_accum', 'entry_price': fill_px,
                                        'margin': margin, 'total_margin': accum_short.total_margin,
                                        'num_entries': len(accum_short.entries)
                                    })
                                    last_accum_bar = i
                                break

        # ========================================
        # === 5분봉 다이버전스 진입 (조건 완화) ===
        # ========================================
        if (i - last_5m_entry_bar) >= 12:  # 1시간 쿨다운

            # === 5분봉 롱: Bullish 다이버전스 (완화된 조건) ===
            # 조건: Bullish Div OR (RSI 과매도 AND Stoch 과매도)
            long_signal = (
                (bullish_div and curr_rsi < futures_cfg.rsi_oversold) or
                (curr_rsi < 35 and curr_stoch_k < 25)  # 강한 과매도
            )

            if position_5m_long is None and long_signal:
                margin = cash * futures_cfg.scalping_pct_5m
                if margin > 0.001 and cash > 0.01:
                    fill_px = apply_costs(c, costs)
                    notional = margin * futures_cfg.leverage
                    size = notional / fill_px

                    position_5m_long = {
                        'entry_bar': i, 'entry_price': fill_px,
                        'size': size, 'margin': margin
                    }
                    cash -= margin

                    trades.append({
                        'time': t, 'type': 'ENTRY', 'side': 'long', 'timeframe': '5m',
                        'entry_price': fill_px, 'rsi': curr_rsi, 'stoch_k': curr_stoch_k,
                        'bullish_div': bullish_div
                    })
                    last_5m_entry_bar = i

            # === 5분봉 숏: Bearish 다이버전스 (완화된 조건) ===
            short_signal = (
                (bearish_div and curr_rsi > futures_cfg.rsi_overbought) or
                (curr_rsi > 65 and curr_stoch_k > 75)  # 강한 과매수
            )

            if position_5m_short is None and short_signal:
                margin = cash * futures_cfg.scalping_pct_5m
                if margin > 0.001 and cash > 0.01:
                    fill_px = apply_costs(c, costs)
                    notional = margin * futures_cfg.leverage
                    size = -notional / fill_px

                    position_5m_short = {
                        'entry_bar': i, 'entry_price': fill_px,
                        'size': size, 'margin': margin
                    }
                    cash -= margin

                    trades.append({
                        'time': t, 'type': 'ENTRY', 'side': 'short', 'timeframe': '5m',
                        'entry_price': fill_px, 'rsi': curr_rsi, 'stoch_k': curr_stoch_k,
                        'bearish_div': bearish_div
                    })
                    last_5m_entry_bar = i

        # === Equity 기록 ===
        equity_val = cash

        # 축적 롱 포지션 미실현 손익
        if accum_long is not None and accum_long.total_size > 0:
            unrealized_pnl = (c - accum_long.avg_entry_price) * accum_long.total_size
            equity_val += accum_long.total_margin + unrealized_pnl

        # 축적 숏 포지션 미실현 손익
        if accum_short is not None and accum_short.total_size < 0:
            unrealized_pnl = (accum_short.avg_entry_price - c) * abs(accum_short.total_size)
            equity_val += accum_short.total_margin + unrealized_pnl

        # 5분봉 포지션 미실현 손익
        if position_5m_long is not None:
            unrealized_pnl = (c - position_5m_long['entry_price']) * position_5m_long['size']
            equity_val += position_5m_long['margin'] + unrealized_pnl

        if position_5m_short is not None:
            unrealized_pnl = (position_5m_short['entry_price'] - c) * abs(position_5m_short['size'])
            equity_val += position_5m_short['margin'] + unrealized_pnl

        equity_rows.append({'time': t, 'equity': equity_val})

    # DataFrame 생성
    equity_df = pd.DataFrame(equity_rows).set_index('time')
    equity_df['ret'] = equity_df['equity'].pct_change().fillna(0)

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # 수익률 계산
    if len(equity_df) > 0:
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity * 100

        # MDD 계산
        running_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - running_max) / running_max * 100
        max_drawdown = drawdown.min()
    else:
        total_return = 0
        max_drawdown = 0

    # 거래 통계
    if len(trades_df) > 0:
        close_types = ['TP', 'SL', 'TIME_EXIT_6H', 'TIME_EXIT_3H', 'LIQUIDATION']
        close_trades = trades_df[trades_df['type'].isin(close_types)]
        if len(close_trades) > 0 and 'pnl' in close_trades.columns:
            wins = close_trades[close_trades['pnl'] > 0]
            win_rate = len(wins) / len(close_trades) * 100
        else:
            win_rate = 0

        # 타임프레임별 통계
        trades_15m = trades_df[trades_df['timeframe'] == '15m_accum']
        trades_5m = trades_df[trades_df['timeframe'] == '5m']
        trades_15m_count = len(trades_15m[trades_15m['type'].isin(close_types)])
        trades_5m_count = len(trades_5m[trades_5m['type'].isin(close_types)])
    else:
        win_rate = 0
        trades_15m_count = 0
        trades_5m_count = 0

    stats = {
        'initial_equity': initial_equity,
        'final_equity': equity_df['equity'].iloc[-1] if len(equity_df) > 0 else initial_equity,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'total_trades': trades_15m_count + trades_5m_count,
        'trades_15m': trades_15m_count,
        'trades_5m': trades_5m_count,
        'accumulation_count': accumulation_count,
        'win_rate': win_rate,
        'liquidations_15m': liquidation_count_15m,
        'liquidations_5m': liquidation_count_5m,
        'total_funding_paid': total_funding_paid,
        'leverage': futures_cfg.leverage
    }

    return equity_df, trades_df, stats
