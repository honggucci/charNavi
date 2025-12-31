"""
선물 백테스트 엔진
Futures Backtest Engine

듀얼 타임프레임 전략:
- 15분봉: Wyckoff 페이즈 기반 축적 매매
- 5분봉: RSI 다이버전스 기반 스캘핑 매매

청산 로직, 펀딩비 포함
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from wpcn.core.types import Theta, BacktestCosts, BacktestConfig
from wpcn.execution.cost import apply_costs
from wpcn.features.indicators import atr, rsi, detect_rsi_divergence, stoch_rsi
from wpcn.wyckoff.phases import detect_wyckoff_phase, DIRECTION_RATIOS, FIB_LEVELS
from wpcn.wyckoff.box import box_engine_freeze


@dataclass
class FuturesConfig:
    """선물 백테스트 설정"""
    leverage: float = 10.0
    margin_mode: str = 'isolated'  # 'isolated' or 'cross'
    maintenance_margin_rate: float = 0.005  # 0.5%
    funding_rate: float = 0.0001  # 0.01% per 8 hours
    funding_interval_bars: int = 32  # 15분봉 기준 8시간 = 32봉

    # 포지션 사이징 - 15분봉
    accumulation_pct_15m: float = 0.01  # 15분봉 축적 시 1%
    phase_c_pct: float = 0.05  # Phase C 시 5%

    # 포지션 사이징 - 5분봉
    scalping_pct_5m: float = 0.01  # 5분봉 스캘핑 시 1%

    # TP/SL - 15분봉 (레버리지 적용 전 기준)
    tp_pct_15m: float = 0.015  # +1.5%
    sl_pct_15m: float = 0.010  # -1.0%

    # TP/SL - 5분봉 (더 타이트)
    tp_pct_5m: float = 0.008  # +0.8%
    sl_pct_5m: float = 0.005  # -0.5%

    # 5분봉 다이버전스 필터
    use_5m_divergence: bool = True
    divergence_lookback: int = 20


@dataclass
class FuturesPosition:
    """선물 포지션"""
    entry_time: Any
    entry_bar: int
    entry_price: float
    size: float  # 양수=롱, 음수=숏
    margin: float
    leverage: float
    liquidation_price: float
    tp_price: float
    sl_price: float
    event: str = ''
    timeframe: str = '15m'


def resample_15m_to_5m_index(df_15m: pd.DataFrame, df_5m: pd.DataFrame) -> pd.DataFrame:
    """15분봉 데이터를 5분봉 인덱스에 맞춰 forward fill"""
    # 15분봉 데이터를 5분봉 타임스탬프에 맞춰 확장
    df_aligned = df_15m.reindex(df_5m.index, method='ffill')
    return df_aligned


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


def calculate_liquidation_price(
    entry_price: float,
    side: str,
    leverage: float,
    maintenance_margin_rate: float = 0.005
) -> float:
    """
    청산가 계산 (격리 마진)

    롱: 청산가 = 진입가 × (1 - 1/레버리지 + 유지마진율)
    숏: 청산가 = 진입가 × (1 + 1/레버리지 - 유지마진율)
    """
    if side == 'long':
        liq_price = entry_price * (1 - 1/leverage + maintenance_margin_rate)
    else:
        liq_price = entry_price * (1 + 1/leverage - maintenance_margin_rate)
    return liq_price


def simulate_futures(
    df_5m: pd.DataFrame,
    theta: Theta,
    costs: BacktestCosts,
    bt_cfg: BacktestConfig,
    futures_cfg: FuturesConfig,
    initial_equity: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    듀얼 타임프레임 선물 백테스트 시뮬레이션

    15분봉: Wyckoff 페이즈 기반 축적 매매 (독립 포지션)
    5분봉: RSI 다이버전스 기반 스캘핑 매매 (독립 포지션)

    Args:
        df_5m: 5분봉 OHLCV 데이터
        theta: 전략 파라미터
        costs: 거래 비용
        bt_cfg: 백테스트 설정
        futures_cfg: 선물 설정
        initial_equity: 초기 자본

    Returns:
        equity_df: 자산 히스토리
        trades_df: 거래 히스토리
        stats: 통계
    """
    # 5분봉 -> 15분봉 리샘플링
    df_15m = resample_5m_to_15m(df_5m)

    # 15분봉 Wyckoff 페이즈 감지
    phases_df = detect_wyckoff_phase(df_15m, theta)

    # 15분봉 박스 감지
    box_df = box_engine_freeze(df_15m, theta)

    # 15분봉 ATR
    atr_15m = atr(df_15m, theta.atr_len)

    # 5분봉 다이버전스 감지
    div_5m = detect_5m_divergence(df_5m, futures_cfg.divergence_lookback)

    # 5분봉 RSI
    rsi_5m = rsi(df_5m['close'], 14)

    # 5분봉 Stochastic RSI
    stoch_5m = stoch_rsi(df_5m['close'], 14, 14, 3, 3)

    # 5분봉 ATR
    atr_5m = atr(df_5m, 14)

    # 상태 변수
    cash = initial_equity

    # === 15분봉 포지션 (독립) ===
    position_15m: Optional[FuturesPosition] = None
    last_entry_bar_15m = -20
    cooldown_15m = 4  # 15분봉 4봉 = 1시간

    # === 5분봉 포지션 (독립) ===
    position_5m: Optional[FuturesPosition] = None
    last_entry_bar_5m = -20
    cooldown_5m = 12  # 5분봉 12봉 = 1시간

    trades: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    # 펀딩비 카운터 (5분봉 기준: 8시간 = 96봉)
    funding_interval_5m = futures_cfg.funding_interval_bars * 3  # 15분봉 32 -> 5분봉 96
    bars_since_funding = 0
    total_funding_paid = 0.0

    # 청산 횟수
    liquidation_count_15m = 0
    liquidation_count_5m = 0

    # 15분봉 바 추적 (3봉마다 15분봉 신호 체크)
    last_15m_bar = None

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
        curr_atr_5m = atr_5m.iloc[i] if i < len(atr_5m) else c * 0.01

        # === 펀딩비 처리 ===
        bars_since_funding += 1
        if bars_since_funding >= funding_interval_5m:
            # 15분봉 포지션 펀딩비
            if position_15m is not None:
                position_value = abs(position_15m.size) * c
                if position_15m.size > 0:
                    funding_fee = position_value * futures_cfg.funding_rate
                else:
                    funding_fee = -position_value * futures_cfg.funding_rate
                cash -= funding_fee
                total_funding_paid += funding_fee

            # 5분봉 포지션 펀딩비
            if position_5m is not None:
                position_value = abs(position_5m.size) * c
                if position_5m.size > 0:
                    funding_fee = position_value * futures_cfg.funding_rate
                else:
                    funding_fee = -position_value * futures_cfg.funding_rate
                cash -= funding_fee
                total_funding_paid += funding_fee

            bars_since_funding = 0

        # ========================================
        # === 15분봉 포지션 관리 ===
        # ========================================
        if position_15m is not None and i > position_15m.entry_bar:
            closed_15m = False

            # 1. 청산 체크
            if position_15m.size > 0 and l <= position_15m.liquidation_price:
                trades.append({
                    'time': t, 'type': 'LIQUIDATION', 'side': 'long', 'timeframe': '15m',
                    'entry_price': position_15m.entry_price,
                    'exit_price': position_15m.liquidation_price,
                    'pnl': -position_15m.margin, 'pnl_pct': -100.0,
                    'leverage': position_15m.leverage, 'event': position_15m.event
                })
                position_15m = None
                liquidation_count_15m += 1
                closed_15m = True

            elif position_15m is not None and position_15m.size < 0 and h >= position_15m.liquidation_price:
                trades.append({
                    'time': t, 'type': 'LIQUIDATION', 'side': 'short', 'timeframe': '15m',
                    'entry_price': position_15m.entry_price,
                    'exit_price': position_15m.liquidation_price,
                    'pnl': -position_15m.margin, 'pnl_pct': -100.0,
                    'leverage': position_15m.leverage, 'event': position_15m.event
                })
                position_15m = None
                liquidation_count_15m += 1
                closed_15m = True

            # 2. 손절 체크
            if not closed_15m and position_15m is not None:
                if position_15m.size > 0 and l <= position_15m.sl_price:
                    exit_px = apply_costs(position_15m.sl_price, costs)
                    pnl = (exit_px - position_15m.entry_price) * position_15m.size
                    pnl_pct = pnl / position_15m.margin * 100
                    cash += position_15m.margin + pnl
                    trades.append({
                        'time': t, 'type': 'SL', 'side': 'long', 'timeframe': '15m',
                        'entry_price': position_15m.entry_price, 'exit_price': exit_px,
                        'pnl': pnl, 'pnl_pct': pnl_pct,
                        'leverage': position_15m.leverage, 'event': position_15m.event
                    })
                    position_15m = None
                    closed_15m = True

                elif position_15m is not None and position_15m.size < 0 and h >= position_15m.sl_price:
                    exit_px = apply_costs(position_15m.sl_price, costs)
                    pnl = (position_15m.entry_price - exit_px) * abs(position_15m.size)
                    pnl_pct = pnl / position_15m.margin * 100
                    cash += position_15m.margin + pnl
                    trades.append({
                        'time': t, 'type': 'SL', 'side': 'short', 'timeframe': '15m',
                        'entry_price': position_15m.entry_price, 'exit_price': exit_px,
                        'pnl': pnl, 'pnl_pct': pnl_pct,
                        'leverage': position_15m.leverage, 'event': position_15m.event
                    })
                    position_15m = None
                    closed_15m = True

            # 3. 익절 체크
            if not closed_15m and position_15m is not None:
                if position_15m.size > 0 and h >= position_15m.tp_price:
                    exit_px = apply_costs(position_15m.tp_price, costs)
                    pnl = (exit_px - position_15m.entry_price) * position_15m.size
                    pnl_pct = pnl / position_15m.margin * 100
                    cash += position_15m.margin + pnl
                    trades.append({
                        'time': t, 'type': 'TP', 'side': 'long', 'timeframe': '15m',
                        'entry_price': position_15m.entry_price, 'exit_price': exit_px,
                        'pnl': pnl, 'pnl_pct': pnl_pct,
                        'leverage': position_15m.leverage, 'event': position_15m.event
                    })
                    position_15m = None
                    closed_15m = True

                elif position_15m is not None and position_15m.size < 0 and l <= position_15m.tp_price:
                    exit_px = apply_costs(position_15m.tp_price, costs)
                    pnl = (position_15m.entry_price - exit_px) * abs(position_15m.size)
                    pnl_pct = pnl / position_15m.margin * 100
                    cash += position_15m.margin + pnl
                    trades.append({
                        'time': t, 'type': 'TP', 'side': 'short', 'timeframe': '15m',
                        'entry_price': position_15m.entry_price, 'exit_price': exit_px,
                        'pnl': pnl, 'pnl_pct': pnl_pct,
                        'leverage': position_15m.leverage, 'event': position_15m.event
                    })
                    position_15m = None
                    closed_15m = True

            # 4. 시간 청산 (15분봉 30봉 = 7.5시간 = 5분봉 90봉)
            if not closed_15m and position_15m is not None and (i - position_15m.entry_bar) >= 90:
                exit_px = apply_costs(c, costs)
                if position_15m.size > 0:
                    pnl = (exit_px - position_15m.entry_price) * position_15m.size
                else:
                    pnl = (position_15m.entry_price - exit_px) * abs(position_15m.size)
                pnl_pct = pnl / position_15m.margin * 100
                cash += position_15m.margin + pnl
                trades.append({
                    'time': t, 'type': 'TIME_EXIT', 'side': 'long' if position_15m.size > 0 else 'short',
                    'timeframe': '15m', 'entry_price': position_15m.entry_price, 'exit_price': exit_px,
                    'pnl': pnl, 'pnl_pct': pnl_pct,
                    'leverage': position_15m.leverage, 'event': position_15m.event
                })
                position_15m = None

        # ========================================
        # === 5분봉 포지션 관리 ===
        # ========================================
        if position_5m is not None and i > position_5m.entry_bar:
            closed_5m = False

            # 1. 청산 체크
            if position_5m.size > 0 and l <= position_5m.liquidation_price:
                trades.append({
                    'time': t, 'type': 'LIQUIDATION', 'side': 'long', 'timeframe': '5m',
                    'entry_price': position_5m.entry_price,
                    'exit_price': position_5m.liquidation_price,
                    'pnl': -position_5m.margin, 'pnl_pct': -100.0,
                    'leverage': position_5m.leverage, 'event': position_5m.event
                })
                position_5m = None
                liquidation_count_5m += 1
                closed_5m = True

            elif position_5m is not None and position_5m.size < 0 and h >= position_5m.liquidation_price:
                trades.append({
                    'time': t, 'type': 'LIQUIDATION', 'side': 'short', 'timeframe': '5m',
                    'entry_price': position_5m.entry_price,
                    'exit_price': position_5m.liquidation_price,
                    'pnl': -position_5m.margin, 'pnl_pct': -100.0,
                    'leverage': position_5m.leverage, 'event': position_5m.event
                })
                position_5m = None
                liquidation_count_5m += 1
                closed_5m = True

            # 2. 손절 체크
            if not closed_5m and position_5m is not None:
                if position_5m.size > 0 and l <= position_5m.sl_price:
                    exit_px = apply_costs(position_5m.sl_price, costs)
                    pnl = (exit_px - position_5m.entry_price) * position_5m.size
                    pnl_pct = pnl / position_5m.margin * 100
                    cash += position_5m.margin + pnl
                    trades.append({
                        'time': t, 'type': 'SL', 'side': 'long', 'timeframe': '5m',
                        'entry_price': position_5m.entry_price, 'exit_price': exit_px,
                        'pnl': pnl, 'pnl_pct': pnl_pct,
                        'leverage': position_5m.leverage, 'event': position_5m.event
                    })
                    position_5m = None
                    closed_5m = True

                elif position_5m is not None and position_5m.size < 0 and h >= position_5m.sl_price:
                    exit_px = apply_costs(position_5m.sl_price, costs)
                    pnl = (position_5m.entry_price - exit_px) * abs(position_5m.size)
                    pnl_pct = pnl / position_5m.margin * 100
                    cash += position_5m.margin + pnl
                    trades.append({
                        'time': t, 'type': 'SL', 'side': 'short', 'timeframe': '5m',
                        'entry_price': position_5m.entry_price, 'exit_price': exit_px,
                        'pnl': pnl, 'pnl_pct': pnl_pct,
                        'leverage': position_5m.leverage, 'event': position_5m.event
                    })
                    position_5m = None
                    closed_5m = True

            # 3. 익절 체크
            if not closed_5m and position_5m is not None:
                if position_5m.size > 0 and h >= position_5m.tp_price:
                    exit_px = apply_costs(position_5m.tp_price, costs)
                    pnl = (exit_px - position_5m.entry_price) * position_5m.size
                    pnl_pct = pnl / position_5m.margin * 100
                    cash += position_5m.margin + pnl
                    trades.append({
                        'time': t, 'type': 'TP', 'side': 'long', 'timeframe': '5m',
                        'entry_price': position_5m.entry_price, 'exit_price': exit_px,
                        'pnl': pnl, 'pnl_pct': pnl_pct,
                        'leverage': position_5m.leverage, 'event': position_5m.event
                    })
                    position_5m = None
                    closed_5m = True

                elif position_5m is not None and position_5m.size < 0 and l <= position_5m.tp_price:
                    exit_px = apply_costs(position_5m.tp_price, costs)
                    pnl = (position_5m.entry_price - exit_px) * abs(position_5m.size)
                    pnl_pct = pnl / position_5m.margin * 100
                    cash += position_5m.margin + pnl
                    trades.append({
                        'time': t, 'type': 'TP', 'side': 'short', 'timeframe': '5m',
                        'entry_price': position_5m.entry_price, 'exit_price': exit_px,
                        'pnl': pnl, 'pnl_pct': pnl_pct,
                        'leverage': position_5m.leverage, 'event': position_5m.event
                    })
                    position_5m = None
                    closed_5m = True

            # 4. 시간 청산 (5분봉 30봉 = 2.5시간)
            if not closed_5m and position_5m is not None and (i - position_5m.entry_bar) >= 30:
                exit_px = apply_costs(c, costs)
                if position_5m.size > 0:
                    pnl = (exit_px - position_5m.entry_price) * position_5m.size
                else:
                    pnl = (position_5m.entry_price - exit_px) * abs(position_5m.size)
                pnl_pct = pnl / position_5m.margin * 100
                cash += position_5m.margin + pnl
                trades.append({
                    'time': t, 'type': 'TIME_EXIT', 'side': 'long' if position_5m.size > 0 else 'short',
                    'timeframe': '5m', 'entry_price': position_5m.entry_price, 'exit_price': exit_px,
                    'pnl': pnl, 'pnl_pct': pnl_pct,
                    'leverage': position_5m.leverage, 'event': position_5m.event
                })
                position_5m = None

        # ========================================
        # === 15분봉 신규 진입 (3봉마다 = 15분봉 마감 시점) ===
        # ========================================
        is_15m_close = (i % 3 == 2)  # 5분봉 3번째 = 15분봉 마감

        if is_15m_close and position_15m is None and (i - last_entry_bar_15m) >= (cooldown_15m * 3):
            if not np.isnan(box_low) and box_width > 0:
                ratios = DIRECTION_RATIOS.get(direction, DIRECTION_RATIOS['unknown'])
                box_pos = (c - box_low) / box_width

                # === 15분봉 롱 진입: Wyckoff 페이즈 기반 ===
                long_phase_ok = direction in ['accumulation', 're_accumulation']
                long_box_ok = box_pos < 0.5

                if long_phase_ok and long_box_ok:
                    for fib_name, fib_pct in [('fib_236', 0.236), ('fib_382', 0.382), ('fib_500', 0.500)]:
                        fib_price = box_low + box_width * fib_pct
                        if abs(c - fib_price) / c < 0.008:  # 0.8% 이내
                            position_pct = futures_cfg.accumulation_pct_15m * ratios['long']
                            if position_pct > 0.001 and cash > 0.01:
                                margin = cash * position_pct
                                fill_px = apply_costs(c, costs)
                                notional = margin * futures_cfg.leverage
                                size = notional / fill_px

                                tp_price = fill_px * (1 + futures_cfg.tp_pct_15m)
                                sl_price = fill_px * (1 - futures_cfg.sl_pct_15m)
                                liq_price = calculate_liquidation_price(
                                    fill_px, 'long', futures_cfg.leverage,
                                    futures_cfg.maintenance_margin_rate
                                )

                                cash -= margin
                                position_15m = FuturesPosition(
                                    entry_time=t, entry_bar=i, entry_price=fill_px,
                                    size=size, margin=margin, leverage=futures_cfg.leverage,
                                    liquidation_price=liq_price, tp_price=tp_price, sl_price=sl_price,
                                    event=f'15m_{fib_name}_long', timeframe='15m'
                                )
                                trades.append({
                                    'time': t, 'type': 'ENTRY', 'side': 'long', 'timeframe': '15m',
                                    'entry_price': fill_px, 'exit_price': 0, 'pnl': 0, 'pnl_pct': 0,
                                    'leverage': futures_cfg.leverage, 'event': f'15m_{fib_name}_long'
                                })
                                last_entry_bar_15m = i
                            break

                # === 15분봉 숏 진입: Wyckoff 페이즈 기반 ===
                short_phase_ok = direction in ['distribution', 're_distribution']
                short_box_ok = box_pos > 0.5

                if position_15m is None and short_phase_ok and short_box_ok:
                    for fib_name, fib_pct in [('fib_618', 0.618), ('fib_786', 0.786)]:
                        fib_price = box_low + box_width * fib_pct
                        if abs(c - fib_price) / c < 0.008:
                            position_pct = futures_cfg.accumulation_pct_15m * ratios['short']
                            if position_pct > 0.001 and cash > 0.01:
                                margin = cash * position_pct
                                fill_px = apply_costs(c, costs)
                                notional = margin * futures_cfg.leverage
                                size = -notional / fill_px

                                tp_price = fill_px * (1 - futures_cfg.tp_pct_15m)
                                sl_price = fill_px * (1 + futures_cfg.sl_pct_15m)
                                liq_price = calculate_liquidation_price(
                                    fill_px, 'short', futures_cfg.leverage,
                                    futures_cfg.maintenance_margin_rate
                                )

                                cash -= margin
                                position_15m = FuturesPosition(
                                    entry_time=t, entry_bar=i, entry_price=fill_px,
                                    size=size, margin=margin, leverage=futures_cfg.leverage,
                                    liquidation_price=liq_price, tp_price=tp_price, sl_price=sl_price,
                                    event=f'15m_{fib_name}_short', timeframe='15m'
                                )
                                trades.append({
                                    'time': t, 'type': 'ENTRY', 'side': 'short', 'timeframe': '15m',
                                    'entry_price': fill_px, 'exit_price': 0, 'pnl': 0, 'pnl_pct': 0,
                                    'leverage': futures_cfg.leverage, 'event': f'15m_{fib_name}_short'
                                })
                                last_entry_bar_15m = i
                            break

        # ========================================
        # === 5분봉 신규 진입 (RSI 다이버전스 기반) ===
        # ========================================
        if position_5m is None and (i - last_entry_bar_5m) >= cooldown_5m:
            # === 5분봉 롱: Bullish 다이버전스 + RSI 과매도 ===
            if bullish_div and curr_rsi < 40 and curr_stoch_k < 30:
                position_pct = futures_cfg.scalping_pct_5m
                if position_pct > 0.001 and cash > 0.01:
                    margin = cash * position_pct
                    fill_px = apply_costs(c, costs)
                    notional = margin * futures_cfg.leverage
                    size = notional / fill_px

                    tp_price = fill_px * (1 + futures_cfg.tp_pct_5m)
                    sl_price = fill_px * (1 - futures_cfg.sl_pct_5m)
                    liq_price = calculate_liquidation_price(
                        fill_px, 'long', futures_cfg.leverage,
                        futures_cfg.maintenance_margin_rate
                    )

                    cash -= margin
                    position_5m = FuturesPosition(
                        entry_time=t, entry_bar=i, entry_price=fill_px,
                        size=size, margin=margin, leverage=futures_cfg.leverage,
                        liquidation_price=liq_price, tp_price=tp_price, sl_price=sl_price,
                        event='5m_bullish_div_long', timeframe='5m'
                    )
                    trades.append({
                        'time': t, 'type': 'ENTRY', 'side': 'long', 'timeframe': '5m',
                        'entry_price': fill_px, 'exit_price': 0, 'pnl': 0, 'pnl_pct': 0,
                        'leverage': futures_cfg.leverage, 'event': '5m_bullish_div_long',
                        'rsi': curr_rsi, 'stoch_k': curr_stoch_k
                    })
                    last_entry_bar_5m = i

            # === 5분봉 숏: Bearish 다이버전스 + RSI 과매수 ===
            elif bearish_div and curr_rsi > 60 and curr_stoch_k > 70:
                position_pct = futures_cfg.scalping_pct_5m
                if position_pct > 0.001 and cash > 0.01:
                    margin = cash * position_pct
                    fill_px = apply_costs(c, costs)
                    notional = margin * futures_cfg.leverage
                    size = -notional / fill_px

                    tp_price = fill_px * (1 - futures_cfg.tp_pct_5m)
                    sl_price = fill_px * (1 + futures_cfg.sl_pct_5m)
                    liq_price = calculate_liquidation_price(
                        fill_px, 'short', futures_cfg.leverage,
                        futures_cfg.maintenance_margin_rate
                    )

                    cash -= margin
                    position_5m = FuturesPosition(
                        entry_time=t, entry_bar=i, entry_price=fill_px,
                        size=size, margin=margin, leverage=futures_cfg.leverage,
                        liquidation_price=liq_price, tp_price=tp_price, sl_price=sl_price,
                        event='5m_bearish_div_short', timeframe='5m'
                    )
                    trades.append({
                        'time': t, 'type': 'ENTRY', 'side': 'short', 'timeframe': '5m',
                        'entry_price': fill_px, 'exit_price': 0, 'pnl': 0, 'pnl_pct': 0,
                        'leverage': futures_cfg.leverage, 'event': '5m_bearish_div_short',
                        'rsi': curr_rsi, 'stoch_k': curr_stoch_k
                    })
                    last_entry_bar_5m = i

        # === Equity 기록 ===
        equity_val = cash

        # 15분봉 포지션 미실현 손익
        if position_15m is not None:
            if position_15m.size > 0:
                unrealized_pnl = (c - position_15m.entry_price) * position_15m.size
            else:
                unrealized_pnl = (position_15m.entry_price - c) * abs(position_15m.size)
            equity_val += position_15m.margin + unrealized_pnl

        # 5분봉 포지션 미실현 손익
        if position_5m is not None:
            if position_5m.size > 0:
                unrealized_pnl = (c - position_5m.entry_price) * position_5m.size
            else:
                unrealized_pnl = (position_5m.entry_price - c) * abs(position_5m.size)
            equity_val += position_5m.margin + unrealized_pnl

        equity_rows.append({
            'time': t,
            'equity': equity_val,
            'position_15m': 1 if (position_15m and position_15m.size > 0) else (-1 if (position_15m and position_15m.size < 0) else 0),
            'position_5m': 1 if (position_5m and position_5m.size > 0) else (-1 if (position_5m and position_5m.size < 0) else 0)
        })

    # DataFrame 생성
    equity_df = pd.DataFrame(equity_rows).set_index('time')
    equity_df['ret'] = equity_df['equity'].pct_change().fillna(0)

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # 통계 계산
    if len(equity_df) > 0:
        total_return = (equity_df['equity'].iloc[-1] - initial_equity) / initial_equity * 100
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()
    else:
        total_return = 0
        max_drawdown = 0

    # 거래 통계
    if len(trades_df) > 0:
        close_trades = trades_df[trades_df['type'].isin(['TP', 'SL', 'TIME_EXIT', 'LIQUIDATION'])]
        wins = close_trades[close_trades['pnl'] > 0]
        losses = close_trades[close_trades['pnl'] <= 0]
        win_rate = len(wins) / len(close_trades) * 100 if len(close_trades) > 0 else 0

        # 타임프레임별 통계
        if 'timeframe' in close_trades.columns:
            trades_15m = close_trades[close_trades['timeframe'] == '15m']
            trades_5m = close_trades[close_trades['timeframe'] == '5m']
            trades_15m_count = len(trades_15m)
            trades_5m_count = len(trades_5m)
        else:
            trades_15m_count = 0
            trades_5m_count = len(close_trades)
    else:
        close_trades = pd.DataFrame()
        win_rate = 0
        trades_15m_count = 0
        trades_5m_count = 0

    stats = {
        'initial_equity': initial_equity,
        'final_equity': equity_df['equity'].iloc[-1] if len(equity_df) > 0 else initial_equity,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'total_trades': len(close_trades) if len(trades_df) > 0 else 0,
        'trades_15m': trades_15m_count,
        'trades_5m': trades_5m_count,
        'win_rate': win_rate,
        'liquidations_15m': liquidation_count_15m,
        'liquidations_5m': liquidation_count_5m,
        'total_funding_paid': total_funding_paid,
        'leverage': futures_cfg.leverage
    }

    return equity_df, trades_df, stats
