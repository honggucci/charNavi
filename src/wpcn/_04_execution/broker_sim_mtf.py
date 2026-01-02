"""
백테스트 시뮬레이션 (MTF 점수 기반 전략) V2

V2 개선사항 (GPT 분석 기반):
1. 리스크 기반 포지션 사이징 (95% all-in → 손절가 기반 사이징)
2. Phase 기반 리스크 관리 (PHASE_RISK_MAP)
3. MFE/MAE 트래킹 (Maximum Favorable/Adverse Excursion)
4. Context/Trigger 점수 분리

Original STRATEGY_ARCHITECTURE.md 기반 MTF 점수 시스템 구현:
- 1W/1D/4H/1H/15M/5M 타임프레임 분석
- 점수 기반 신호 생성 (최소 4.0점, TF 정렬 >= 2)
- Navigation Gate 통합
- 손익비 (RR >= 1.5) 체크
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd

import os
from wpcn._03_common._01_core.types import Theta, BacktestCosts, BacktestConfig
from wpcn._04_execution.cost import apply_costs
from wpcn._06_engine.navigation import compute_navigation
from wpcn._03_common._04_navigation.mtf_scoring import (
    compute_mtf_scores,
    generate_mtf_signals,
    calc_confidence_mult
)
from wpcn._03_common._02_features.indicators import atr as calc_atr

# 확률 모델 (조건부 import)
try:
    from wpcn._05_probability.barrier import calculate_barrier_probability
    PROBABILITY_MODEL_AVAILABLE = True
except ImportError:
    PROBABILITY_MODEL_AVAILABLE = False

# 환경변수에서 확률 필터 설정 로드
USE_PROBABILITY_MODEL = os.getenv('USE_PROBABILITY_MODEL', 'False').lower() == 'true'
MIN_TP_PROBABILITY = float(os.getenv('MIN_TP_PROBABILITY', '0.50'))
MIN_EV_R = float(os.getenv('MIN_EV_R', '0.05'))


def calc_qty_risk_based(
    cash: float,
    entry_px: float,
    stop_px: float,
    risk_pct: float,
    confidence_mult: float = 1.0,
    fee_rate: float = 0.0002,
    slippage_rate: float = 0.0002,
    max_alloc_pct: float = 0.95,
) -> float:
    """
    리스크 기반 포지션 사이징

    계산 공식:
    - risk_per_unit = |entry - stop| + 비용
    - risk_amount = cash * risk_pct * confidence_mult
    - qty = risk_amount / risk_per_unit

    Args:
        cash: 현재 현금
        entry_px: 진입가
        stop_px: 손절가
        risk_pct: 리스크 비율 (예: 0.02 = 2%)
        confidence_mult: 신뢰도 배수 (0.5 ~ 1.5)
        fee_rate: 수수료 비율
        slippage_rate: 슬리피지 비율
        max_alloc_pct: 최대 할당 비율 (안전장치)

    Returns:
        수량 (qty)
    """
    if entry_px <= 0 or cash <= 0:
        return 0.0

    # 단위당 리스크 = 진입가 - 손절가 (절대값)
    risk_per_unit = abs(entry_px - stop_px)
    if risk_per_unit <= 0:
        return 0.0

    # 비용 반영 (진입 + 청산 시 비용)
    cost_per_unit = entry_px * (fee_rate + slippage_rate) + stop_px * (fee_rate + slippage_rate)
    effective_risk_per_unit = risk_per_unit + cost_per_unit

    # 허용 손실액 = 현금 * 리스크% * 신뢰도 배수
    risk_amount = cash * risk_pct * confidence_mult

    # 리스크 기반 수량
    qty_risk = risk_amount / effective_risk_per_unit

    # 최대 할당 제한 (안전장치)
    max_alloc = cash * max_alloc_pct
    qty_cash_cap = max_alloc / (entry_px * (1 + fee_rate + slippage_rate))

    qty = min(qty_risk, qty_cash_cap)
    return max(qty, 0.0)


def simulate_mtf(
    df: pd.DataFrame,
    theta: Theta,
    costs: BacktestCosts,
    cfg: BacktestConfig,
    mtf: List[str] | None = None,
    spot_mode: bool = True,
    min_score: float = 4.0,
    min_tf_alignment: int = 2,
    min_rr_ratio: float = 1.5,
    sl_atr_mult: float = 2.0,
    tp_atr_mult: float = 4.0,
    atr_period: int = 14,
    min_context_score: float = 2.0,
    min_trigger_score: float = 1.5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    MTF 점수 기반 백테스트 시뮬레이션

    Args:
        df: 5M OHLCV 데이터
        theta: Wyckoff 감지 파라미터
        costs: 거래 비용
        cfg: 백테스트 설정
        mtf: 추가 타임프레임 리스트 (기본: ['15m', '1h', '4h', '1d', '1w'])
        spot_mode: True면 현물 (숏 비활성화), False면 선물
        min_score: 최소 신호 점수 (기본: 4.0)
        min_tf_alignment: 최소 TF 정렬 개수 (기본: 2)
        min_rr_ratio: 최소 손익비 (기본: 1.5)
        sl_atr_mult: SL ATR 배수 (기본: 2.0)
        tp_atr_mult: TP ATR 배수 (기본: 4.0)
        atr_period: ATR 기간 (기본: 14)

    Returns:
        (equity_df, trades_df, signals_df, nav_df)
    """
    df = df.sort_index()

    if mtf is None:
        mtf = ['15m', '1h', '4h', '1d', '1w']

    print("  Computing MTF scores...")
    mtf_scores = compute_mtf_scores(df, theta, timeframes=mtf)

    print("  Generating MTF signals...")
    signals_df = generate_mtf_signals(
        mtf_scores,
        df,
        min_score=min_score,
        min_tf_alignment=min_tf_alignment,
        min_rr_ratio=min_rr_ratio,
        atr_period=atr_period,
        sl_atr_mult=sl_atr_mult,
        tp_atr_mult=tp_atr_mult,
        min_context_score=min_context_score,
        min_trigger_score=min_trigger_score
    )

    # 현물 모드: 숏 제거
    if spot_mode:
        signals_df.loc[signals_df['signal'] == 'SHORT', 'signal'] = None

    # Count valid signals
    valid_signals = signals_df['signal'].notna().sum()
    long_signals = (signals_df['signal'] == 'LONG').sum()
    short_signals = (signals_df['signal'] == 'SHORT').sum()
    print(f"  Total signals: {valid_signals} (LONG: {long_signals}, SHORT: {short_signals})")

    # === Debug: Navigation Gate 통과율 분석 ===
    gate_passed = 0
    gate_blocked = 0
    price_hit = 0
    price_miss = 0
    prob_passed = 0
    prob_blocked = 0

    # Navigation gate
    print("  Computing navigation gate...")
    nav = compute_navigation(df, theta, cfg, mtf)
    nav_gate = nav["trade_gate"].to_dict() if not nav.empty else {}

    # Initialize state
    cash = float(cfg.initial_equity)
    pos_side = 0  # +1 long, -1 short, 0 flat
    pos_qty = 0.0
    entry_px = np.nan
    stop_px = np.nan
    tp_px = np.nan
    entry_i = -1
    tp1_done = False

    # === V2: MFE/MAE 트래킹 ===
    mfe = 0.0  # Maximum Favorable Excursion (최대 유리 이동)
    mae = 0.0  # Maximum Adverse Excursion (최대 불리 이동)
    current_phase = 'UNKNOWN'
    current_risk_pct = 0.01

    trades: List[dict] = []
    equity_rows: List[dict] = []

    # Pre-extract arrays
    idx = df.index
    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    close_arr = df["close"].values
    n = len(df)

    for i in range(n):
        t = idx[i]
        o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]

        # === V2: MFE/MAE 업데이트 (포지션 보유 중) ===
        if pos_side != 0:
            if pos_side == 1:  # Long
                unrealized_pct = (c - entry_px) / entry_px * 100
                max_unrealized = (h - entry_px) / entry_px * 100
                min_unrealized = (l - entry_px) / entry_px * 100
                mfe = max(mfe, max_unrealized)
                mae = min(mae, min_unrealized)
            else:  # Short
                unrealized_pct = (entry_px - c) / entry_px * 100
                max_unrealized = (entry_px - l) / entry_px * 100
                min_unrealized = (entry_px - h) / entry_px * 100
                mfe = max(mfe, max_unrealized)
                mae = min(mae, min_unrealized)

        # === Exit logic ===
        if pos_side != 0:
            # Stop loss
            if (pos_side == 1 and l <= stop_px) or (pos_side == -1 and h >= stop_px):
                exit_px = apply_costs(stop_px, costs)
                if pos_side == 1:
                    cash += pos_qty * exit_px
                    pnl_pct = (exit_px - entry_px) / entry_px * 100
                else:  # short
                    pnl = pos_qty * (entry_px - exit_px)
                    cash += pos_qty * entry_px + pnl
                    pnl_pct = (entry_px - exit_px) / entry_px * 100

                trades.append({
                    "time": t,
                    "type": "STOP",
                    "side": "long" if pos_side == 1 else "short",
                    "price": exit_px,
                    "qty": pos_qty,
                    "pnl_pct": pnl_pct,
                    "mfe": mfe,
                    "mae": mae,
                    "phase": current_phase,
                    "risk_pct": current_risk_pct
                })
                pos_qty = 0.0
                pos_side = 0
                entry_px = np.nan
                stop_px = np.nan
                tp_px = np.nan
                entry_i = -1
                tp1_done = False
                mfe = 0.0
                mae = 0.0

            # TP1 (partial exit)
            elif not tp1_done and ((pos_side == 1 and h >= tp_px) or (pos_side == -1 and l <= tp_px)):
                exit_px = apply_costs(tp_px, costs)
                frac = cfg.tp1_frac  # Partial exit fraction
                qty_exit = pos_qty * frac

                if pos_side == 1:
                    cash += qty_exit * exit_px
                    pnl_pct = (exit_px - entry_px) / entry_px * 100
                else:
                    pnl = qty_exit * (entry_px - exit_px)
                    cash += qty_exit * entry_px + pnl
                    pnl_pct = (entry_px - exit_px) / entry_px * 100

                trades.append({
                    "time": t,
                    "type": "TP1",
                    "side": "long" if pos_side == 1 else "short",
                    "price": exit_px,
                    "qty": qty_exit,
                    "pnl_pct": pnl_pct,
                    "mfe": mfe,
                    "mae": mae,
                    "phase": current_phase,
                    "risk_pct": current_risk_pct
                })
                pos_qty *= (1 - frac)
                tp1_done = True

                # Move SL to breakeven after TP1
                stop_px = entry_px

            # TP2 (full exit on remaining position)
            elif tp1_done and cfg.use_tp2 and pos_qty > 0:
                # Use a higher TP for TP2 (e.g., 1.5x the original TP)
                tp2_px = entry_px + (tp_px - entry_px) * 1.5 if pos_side == 1 else entry_px - (entry_px - tp_px) * 1.5

                if (pos_side == 1 and h >= tp2_px) or (pos_side == -1 and l <= tp2_px):
                    exit_px = apply_costs(tp2_px, costs)

                    if pos_side == 1:
                        cash += pos_qty * exit_px
                        pnl_pct = (exit_px - entry_px) / entry_px * 100
                    else:
                        pnl = pos_qty * (entry_px - exit_px)
                        cash += pos_qty * entry_px + pnl
                        pnl_pct = (entry_px - exit_px) / entry_px * 100

                    trades.append({
                        "time": t,
                        "type": "TP2",
                        "side": "long" if pos_side == 1 else "short",
                        "price": exit_px,
                        "qty": pos_qty,
                        "pnl_pct": pnl_pct,
                        "mfe": mfe,
                        "mae": mae,
                        "phase": current_phase,
                        "risk_pct": current_risk_pct
                    })
                    pos_qty = 0.0
                    pos_side = 0
                    entry_px = np.nan
                    stop_px = np.nan
                    tp_px = np.nan
                    entry_i = -1
                    tp1_done = False
                    mfe = 0.0
                    mae = 0.0

            # Max hold time
            elif cfg.max_hold_bars and (i - entry_i) >= cfg.max_hold_bars:
                exit_px = apply_costs(c, costs)

                if pos_side == 1:
                    cash += pos_qty * exit_px
                    pnl_pct = (exit_px - entry_px) / entry_px * 100
                else:
                    pnl = pos_qty * (entry_px - exit_px)
                    cash += pos_qty * entry_px + pnl
                    pnl_pct = (entry_px - exit_px) / entry_px * 100

                trades.append({
                    "time": t,
                    "type": "TIME_EXIT",
                    "side": "long" if pos_side == 1 else "short",
                    "price": exit_px,
                    "qty": pos_qty,
                    "pnl_pct": pnl_pct,
                    "mfe": mfe,
                    "mae": mae,
                    "phase": current_phase,
                    "risk_pct": current_risk_pct
                })
                pos_qty = 0.0
                pos_side = 0
                entry_px = np.nan
                stop_px = np.nan
                tp_px = np.nan
                entry_i = -1
                tp1_done = False
                mfe = 0.0
                mae = 0.0

        # === Entry logic ===
        if pos_side == 0 and t in signals_df.index:
            signal = signals_df.loc[t, 'signal']

            if pd.notna(signal):
                # Check navigation gate
                gate_ok = bool(int(nav_gate.get(t, 1)))

                if not gate_ok:
                    gate_blocked += 1
                else:
                    gate_passed += 1

                if gate_ok:
                    signal_entry = signals_df.loc[t, 'entry_price']
                    signal_sl = signals_df.loc[t, 'stop_loss']
                    signal_tp = signals_df.loc[t, 'take_profit']
                    signal_score = signals_df.loc[t, 'score']
                    signal_rr = signals_df.loc[t, 'rr_ratio']

                    # Check if price hits entry level
                    if l <= signal_entry <= h:
                        price_hit += 1
                    else:
                        price_miss += 1

                    if l <= signal_entry <= h:
                        # === 4.5단계: 확률 필터 (TP 적중 확률 + 기대값 체크) ===
                        prob_filter_ok = True
                        barrier_result = None

                        if USE_PROBABILITY_MODEL and PROBABILITY_MODEL_AVAILABLE:
                            try:
                                # ATR 계산 (변동성 추정용)
                                atr_val = calc_atr(df.iloc[:i+1], period=atr_period).iloc[-1] if i >= atr_period else c * 0.02

                                # 롱/숏 방향 결정
                                is_long = (signal == 'LONG')

                                # 수수료/슬리피지 비율
                                fee_rate = costs.fee_bps / 10000
                                slippage_rate = costs.slippage_bps / 10000

                                # 배리어 확률 계산 (barrier.py의 실제 시그니처 사용)
                                barrier_result = calculate_barrier_probability(
                                    current_price=c,
                                    tp_price=signal_tp,
                                    sl_price=signal_sl,
                                    atr=atr_val,
                                    max_hold_bars=cfg.max_hold_bars if cfg.max_hold_bars else 48,
                                    is_long=is_long,
                                    entry_fee_pct=fee_rate,
                                    exit_fee_pct=fee_rate,
                                    slippage_pct=slippage_rate,
                                    use_monte_carlo=True,
                                    n_simulations=1000  # 빠른 시뮬레이션
                                )

                                # 확률 필터 조건 체크
                                # P(TP) >= MIN_TP_PROBABILITY 이고 EV_R >= MIN_EV_R 이어야 진입
                                if barrier_result.p_tp < MIN_TP_PROBABILITY:
                                    prob_filter_ok = False
                                elif barrier_result.ev_r < MIN_EV_R:
                                    prob_filter_ok = False

                            except Exception as e:
                                # 확률 계산 실패 시 필터 통과 (기존 로직 유지)
                                prob_filter_ok = True

                        if not prob_filter_ok:
                            prob_blocked += 1
                            continue  # 확률 조건 미달 - 진입 스킵
                        else:
                            prob_passed += 1

                        fill_px = apply_costs(signal_entry, costs)

                        # === V2: 리스크 기반 포지션 사이징 ===
                        # Phase별 리스크% 사용 (기본 1%)
                        signal_phase = signals_df.loc[t, 'phase'] if 'phase' in signals_df.columns else 'UNKNOWN'
                        signal_risk_pct = signals_df.loc[t, 'risk_pct'] if 'risk_pct' in signals_df.columns else 0.01

                        # Context/Trigger 점수로 confidence 계산
                        signal_context = signals_df.loc[t, 'context_score'] if 'context_score' in signals_df.columns else 2.0
                        signal_trigger = signals_df.loc[t, 'trigger_score'] if 'trigger_score' in signals_df.columns else 1.5
                        confidence_mult = calc_confidence_mult(
                            nav_conf=signal_context / 4.0,  # context를 0~1로 정규화
                            trigger_score=signal_trigger,
                            trigger_min=1.5
                        )

                        # 수수료/슬리피지 비율
                        fee_rate = costs.fee_bps / 10000
                        slippage_rate = costs.slippage_bps / 10000

                        # 리스크 기반 수량 계산
                        qty = calc_qty_risk_based(
                            cash=cash,
                            entry_px=fill_px,
                            stop_px=signal_sl,
                            risk_pct=signal_risk_pct,
                            confidence_mult=confidence_mult,
                            fee_rate=fee_rate,
                            slippage_rate=slippage_rate,
                            max_alloc_pct=0.95
                        )

                        if qty <= 0:
                            continue  # 수량이 0이면 스킵

                        alloc = qty * fill_px

                        if signal == 'LONG':
                            cash -= alloc
                            pos_side = 1
                            pos_qty = qty
                            entry_px = fill_px
                            stop_px = signal_sl
                            tp_px = signal_tp
                            entry_i = i
                            tp1_done = False
                            mfe = 0.0
                            mae = 0.0
                            current_phase = signal_phase
                            current_risk_pct = signal_risk_pct

                            trades.append({
                                "time": t,
                                "type": "ENTRY",
                                "side": "long",
                                "price": fill_px,
                                "qty": qty,
                                "score": signal_score,
                                "rr_ratio": signal_rr,
                                "sl": stop_px,
                                "tp": tp_px,
                                "phase": signal_phase,
                                "risk_pct": signal_risk_pct,
                                "context_score": signal_context,
                                "trigger_score": signal_trigger,
                                "confidence_mult": confidence_mult
                            })

                        elif signal == 'SHORT' and not spot_mode:
                            # Short entry (margin-based)
                            cash -= alloc
                            pos_side = -1
                            pos_qty = qty
                            entry_px = fill_px
                            stop_px = signal_sl
                            tp_px = signal_tp
                            entry_i = i
                            tp1_done = False
                            mfe = 0.0
                            mae = 0.0
                            current_phase = signal_phase
                            current_risk_pct = signal_risk_pct

                            trades.append({
                                "time": t,
                                "type": "ENTRY",
                                "side": "short",
                                "price": fill_px,
                                "qty": qty,
                                "score": signal_score,
                                "rr_ratio": signal_rr,
                                "sl": stop_px,
                                "tp": tp_px,
                                "phase": signal_phase,
                                "risk_pct": signal_risk_pct,
                                "context_score": signal_context,
                                "trigger_score": signal_trigger,
                                "confidence_mult": confidence_mult
                            })

        # === Equity calculation ===
        if pos_side == 1:  # Long
            equity_val = cash + pos_qty * c
        elif pos_side == -1:  # Short
            equity_val = cash + pos_qty * entry_px + pos_qty * (entry_px - c)
        else:  # Flat
            equity_val = cash

        equity_rows.append({
            "time": t,
            "equity": float(equity_val),
            "pos_qty": float(pos_qty),
            "pos_side": int(pos_side)
        })

    # === Debug: 필터링 통계 출력 ===
    print(f"  [Debug] Gate passed: {gate_passed}, Gate blocked: {gate_blocked}")
    print(f"  [Debug] Price hit: {price_hit}, Price miss: {price_miss}")
    if USE_PROBABILITY_MODEL:
        print(f"  [Debug] Prob passed: {prob_passed}, Prob blocked: {prob_blocked}")
    print(f"  [Debug] Total entries: {len([t for t in trades if t.get('type') == 'ENTRY'])}")

    # Build output DataFrames
    equity_df = pd.DataFrame(equity_rows).set_index("time")
    equity_df["ret"] = equity_df["equity"].pct_change().fillna(0.0)

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["time", "type", "side", "price"])

    # Format signals_df for output
    signals_out = signals_df[signals_df['signal'].notna()].copy()
    signals_out = signals_out.rename(columns={
        'signal': 'side',
        'entry_price': 'entry',
        'stop_loss': 'stop',
        'take_profit': 'tp1'
    })
    signals_out['event'] = 'MTF_SCORE'
    signals_out['tp2'] = signals_out['tp1']  # Dummy tp2

    return equity_df, trades_df, signals_out, nav
