from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import os

from wpcn._03_common._01_core.types import Theta, BacktestCosts, BacktestConfig
from wpcn._04_execution.cost import apply_costs
from wpcn._03_common._03_wyckoff.events import detect_spring_utad, detect_all_signals
from wpcn._03_common._03_wyckoff.phases import detect_phase_signals, AccumulationSignal
from wpcn._06_engine.navigation import compute_navigation
from wpcn._03_common._02_features.dynamic_params import DynamicParameterEngine

def approx_pfill_det(df: pd.DataFrame, t_idx: int, entry_price: float, N_fill: int) -> float:
    end = min(len(df), t_idx + 1 + N_fill)
    window = df.iloc[t_idx+1:end]
    if window.empty:
        return 0.0
    hit = ((window["low"] <= entry_price) & (entry_price <= window["high"])).any()
    return 1.0 if hit else 0.0

def simulate(df: pd.DataFrame, theta: Theta, costs: BacktestCosts, cfg: BacktestConfig, mtf: List[str] | None = None, scalping_mode: bool = False, use_phase_accumulation: bool = False, spot_mode: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    백테스트 시뮬레이션 (와이코프 + 피보나치 전략)

    Args:
        spot_mode: True면 현물 (숏 비활성화), False면 선물 (숏 가능)
        use_phase_accumulation: DEPRECATED - 항상 False (313번 물타기 방지)
    """
    df = df.sort_index()
    mtf = mtf or []

    # === 동적 파라미터 계산 (전체 시계열) ===
    print("  Computing dynamic parameters...")
    dyn_engine = DynamicParameterEngine(
        vol_lookback=100,
        fft_min_period=10,
        fft_max_period=200,
        hilbert_smooth=5
    )
    # 전체 시계열에 대한 동적 파라미터 계산 (미리 계산)
    dyn_params_df = dyn_engine.compute_params_series(df, min_lookback=50)
    print(f"  Dynamic parameters computed for {len(dyn_params_df)} bars")

    # navigation (page probs + Unknown gate)
    nav = compute_navigation(df, theta, cfg, mtf)

    # Use scalping signals if scalping_mode is enabled
    signals = detect_all_signals(df, theta, scalping_mode=scalping_mode)

    # Phase detection for box levels (but NOT for accumulation signals)
    phases_df, _, _ = detect_phase_signals(df, theta, accumulation_pct=0.03)

    # 현물 모드: 메인 신호에서도 숏 제거
    if spot_mode:
        signals = [s for s in signals if s.side == 'long']
    signals_df = pd.DataFrame([{
        "time": s.t_signal, "side": s.side, "event": s.event,
        "entry": s.entry_price, "stop": s.stop_price, "tp1": s.tp1, "tp2": s.tp2,
        **{f"meta_{k}": v for k,v in (s.meta or {}).items()}
    } for s in signals]).sort_values("time") if signals else pd.DataFrame(columns=["time"])

    cash = float(cfg.initial_equity)

    # Fibonacci-based position management
    fib_positions: List[Dict[str, Any]] = []  # 피보나치 분할 포지션들
    total_qty = 0.0  # 총 포지션 수량
    avg_entry_px = 0.0  # 평균 진입가
    pos_side = 0  # +1 long, -1 short
    entry_i = -1
    stop_price = np.nan

    # Fibonacci entry tracking
    fib_236_filled = False
    fib_382_filled = False
    fib_500_filled = False

    # Fibonacci exit tracking
    fib_618_exited = False
    fib_786_exited = False

    # Spring/UTAD event tracking
    spring_detected_i = -100  # Spring 발생 바 인덱스
    utad_detected_i = -100  # UTAD 발생 바 인덱스
    active_box_high = np.nan
    active_box_low = np.nan

    pending_orders: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    sig_by_time: Dict[Any, List[Any]] = {}
    for s in signals:
        sig_by_time.setdefault(s.t_signal, []).append(s)

    # OPTIMIZATION: Pre-extract numpy arrays for faster iteration
    idx = df.index
    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    close_arr = df["close"].values

    # Pre-index nav for faster lookup
    nav_gate = nav["trade_gate"].to_dict() if not nav.empty else {}
    n = len(df)

    for i in range(n):
        t = idx[i]
        o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]

        # Get current box levels from phases_df
        if t in phases_df.index:
            current_box_high = phases_df.loc[t, 'box_high']
            current_box_low = phases_df.loc[t, 'box_low']
            box_range = current_box_high - current_box_low

            # Calculate Fibonacci levels
            fib_0 = current_box_low
            fib_236 = current_box_low + box_range * 0.236
            fib_382 = current_box_low + box_range * 0.382
            fib_500 = current_box_low + box_range * 0.5
            fib_618 = current_box_low + box_range * 0.618
            fib_786 = current_box_low + box_range * 0.786
            fib_1 = current_box_high
        else:
            fib_0 = fib_236 = fib_382 = fib_500 = fib_618 = fib_786 = fib_1 = np.nan

        # === Spring/UTAD 감지 ===
        # Spring/UTAD 신호가 발생하면 box level 저장
        if t in sig_by_time:
            for sig in sig_by_time[t]:
                if sig.event in ['spring', 'upthrust'] and sig.side == 'long':
                    spring_detected_i = i
                    active_box_high = current_box_high
                    active_box_low = current_box_low
                elif sig.event in ['utad', 'downthrust'] and sig.side == 'short':
                    utad_detected_i = i
                    active_box_high = current_box_high
                    active_box_low = current_box_low

        # === 피보나치 기반 분할 진입 (Spring 이후 롱) ===
        if pos_side == 0 and spring_detected_i >= 0 and (i - spring_detected_i) < 50:
            # Spring 발생 후 50봉 이내에만 진입 가능
            gate_ok = bool(int(nav_gate.get(t, 1)))
            if gate_ok and not np.isnan(fib_236):
                # Fib 0.236 진입 (30% 자본)
                if not fib_236_filled and l <= fib_236:
                    entry_px = apply_costs(fib_236, costs)
                    qty = (cash * 0.3) / entry_px
                    cash -= qty * entry_px
                    fib_positions.append({"price": entry_px, "qty": qty, "level": "fib_236"})
                    total_qty += qty
                    avg_entry_px = sum(p["price"] * p["qty"] for p in fib_positions) / total_qty
                    pos_side = 1
                    entry_i = i
                    stop_price = active_box_low - (active_box_high - active_box_low) * 0.1
                    fib_236_filled = True
                    trades.append({"time": t, "type": "ENTRY_FIB_236", "side": "long", "price": entry_px, "qty": qty})

                # Fib 0.382 추가 진입 (20% 자본)
                elif fib_236_filled and not fib_382_filled and l <= fib_382:
                    entry_px = apply_costs(fib_382, costs)
                    qty = (cash * 0.2) / entry_px
                    cash -= qty * entry_px
                    fib_positions.append({"price": entry_px, "qty": qty, "level": "fib_382"})
                    total_qty += qty
                    avg_entry_px = sum(p["price"] * p["qty"] for p in fib_positions) / total_qty
                    fib_382_filled = True
                    trades.append({"time": t, "type": "ENTRY_FIB_382", "side": "long", "price": entry_px, "qty": qty})

                # Fib 0.5 마지막 진입 (10% 자본)
                elif fib_382_filled and not fib_500_filled and l <= fib_500:
                    entry_px = apply_costs(fib_500, costs)
                    qty = (cash * 0.1) / entry_px
                    cash -= qty * entry_px
                    fib_positions.append({"price": entry_px, "qty": qty, "level": "fib_500"})
                    total_qty += qty
                    avg_entry_px = sum(p["price"] * p["qty"] for p in fib_positions) / total_qty
                    fib_500_filled = True
                    trades.append({"time": t, "type": "ENTRY_FIB_500", "side": "long", "price": entry_px, "qty": qty})

        # === 피보나치 기반 분할 청산 (롱) ===
        if pos_side == 1 and total_qty > 0:
            # 손절: Spring 저점 아래 이탈
            if l <= stop_price:
                exit_px = apply_costs(stop_price, costs)
                cash += total_qty * exit_px
                trades.append({"time": t, "type": "STOP", "side": "long", "price": exit_px, "qty": total_qty, "pnl_pct": (exit_px - avg_entry_px) / avg_entry_px * 100})
                # 포지션 리셋
                total_qty = 0.0
                avg_entry_px = 0.0
                pos_side = 0
                fib_positions = []
                fib_236_filled = fib_382_filled = fib_500_filled = False
                fib_618_exited = fib_786_exited = False
                spring_detected_i = -100
                continue

            # Fib 0.618 익절 (30% 청산)
            if not fib_618_exited and h >= fib_618 and not np.isnan(fib_618):
                exit_qty = total_qty * 0.3
                exit_px = apply_costs(fib_618, costs)
                cash += exit_qty * exit_px
                total_qty -= exit_qty
                fib_618_exited = True
                # SL을 진입가로 이동 (Break-even)
                stop_price = avg_entry_px
                trades.append({"time": t, "type": "TP_FIB_618", "side": "long", "price": exit_px, "qty": exit_qty, "pnl_pct": (exit_px - avg_entry_px) / avg_entry_px * 100})

            # Fib 0.786 익절 (추가 30% 청산)
            elif fib_618_exited and not fib_786_exited and h >= fib_786 and not np.isnan(fib_786):
                exit_qty = total_qty * 0.4286  # 남은 70%의 30% = 전체의 30%
                exit_px = apply_costs(fib_786, costs)
                cash += exit_qty * exit_px
                total_qty -= exit_qty
                fib_786_exited = True
                trades.append({"time": t, "type": "TP_FIB_786", "side": "long", "price": exit_px, "qty": exit_qty, "pnl_pct": (exit_px - avg_entry_px) / avg_entry_px * 100})

            # Fib 1.0 (Box High) 전체 청산 (남은 40%)
            elif fib_786_exited and h >= fib_1 and not np.isnan(fib_1):
                exit_px = apply_costs(fib_1, costs)
                cash += total_qty * exit_px
                trades.append({"time": t, "type": "TP_FIB_1", "side": "long", "price": exit_px, "qty": total_qty, "pnl_pct": (exit_px - avg_entry_px) / avg_entry_px * 100})
                # 포지션 리셋
                total_qty = 0.0
                avg_entry_px = 0.0
                pos_side = 0
                fib_positions = []
                fib_236_filled = fib_382_filled = fib_500_filled = False
                fib_618_exited = fib_786_exited = False
                spring_detected_i = -100
                continue

            # 최대 보유 기간 초과 시 강제 청산
            if cfg.max_hold_bars and (i - entry_i) >= cfg.max_hold_bars:
                exit_px = apply_costs(c, costs)
                cash += total_qty * exit_px
                trades.append({"time": t, "type": "TIME_EXIT", "side": "long", "price": exit_px, "qty": total_qty, "pnl_pct": (exit_px - avg_entry_px) / avg_entry_px * 100})
                # 포지션 리셋
                total_qty = 0.0
                avg_entry_px = 0.0
                pos_side = 0
                fib_positions = []
                fib_236_filled = fib_382_filled = fib_500_filled = False
                fib_618_exited = fib_786_exited = False
                spring_detected_i = -100
                continue

        # === 기존 pending orders 로직 (레거시, Spring 없을 때 폴백용) ===
        # place new orders (activate next bar) if gate open
        if pos_side == 0 and t in sig_by_time:
            gate_ok = bool(int(nav_gate.get(t, 1)))
            if use_fractal_gate:
                # HTF Phase 확인 (4h 우선, 없으면 1h)
                htf_phase_4h = nav.loc[t, "phase_4h"] if "phase_4h" in nav.columns and t in nav.index else "unknown"
                htf_phase_1h = nav.loc[t, "phase_1h"] if "phase_1h" in nav.columns and t in nav.index else "unknown"

                # Phase C (Markup/Markdown) = 추세장 → Navigation Gate 엄격 적용
                if htf_phase_4h == "C" or (htf_phase_4h == "unknown" and htf_phase_1h == "C"):
                    gate_ok = bool(int(nav_gate.get(t, 1)))  # 원래 Navigation Gate 적용
                # Phase A/B (Accumulation/Distribution) = 횡보장 → Navigation Gate 비활성화
                else:
                    gate_ok = True  # 횡보장에서는 Phase 축적 신호 허용
            else:
                # 기존 로직: Navigation Gate 무조건 적용
                gate_ok = bool(int(nav_gate.get(t, 1)))

            if not gate_ok:
                continue  # Navigation Gate 통과 못하면 축적 신호 무시

            for acc_sig in accum_by_time[t]:
                # === 동적 추세 필터링 ===
                dyn_params = dyn_params_df.loc[t]
                trend_direction = dyn_params['trend_direction']
                trend_strength = dyn_params['trend_strength']

                # 추세 기반 진입 필터링
                # 하락 추세(bearish)일 때 롱 신호 필터링
                # 상승 추세(bullish)일 때 숏 신호 필터링
                # 추세 강도가 0.3 이상일 때 적용 (중간 이상 추세 필터)
                if trend_strength >= 0.3:
                    if acc_sig.side == 'long' and trend_direction == 'bearish':
                        continue  # 하락 추세에서 롱 진입 스킵
                    if acc_sig.side == 'short' and trend_direction == 'bullish':
                        continue  # 상승 추세에서 숏 진입 스킵

                # 축적에 사용할 금액 (현금의 position_pct%)
                alloc_amount = cash * acc_sig.position_pct
                if alloc_amount < 0.0001:  # 최소 금액 체크
                    continue

                fill_px = apply_costs(acc_sig.entry_price, costs)
                qty = alloc_amount / fill_px

                if acc_sig.side == 'long':
                    # === 롱 축적 ===
                    new_total_qty = accum_total_qty + qty
                    if new_total_qty > 0:
                        accum_avg_price = (accum_avg_price * accum_total_qty + fill_px * qty) / new_total_qty
                    accum_total_qty = new_total_qty
                    cash -= alloc_amount
                    accum_stop_price = acc_sig.stop_price
                    accum_tp_price = acc_sig.tp_price
                    accum_last_entry_i = i  # 진입 바 기록

                    accum_positions.append({
                        "time": t, "price": fill_px, "qty": qty,
                        "event": acc_sig.event, "phase": acc_sig.phase
                    })
                    trades.append({
                        "time": t, "type": "ACCUM_LONG", "side": "long",
                        "price": fill_px, "event": acc_sig.event,
                        "phase": acc_sig.phase, "position_pct": acc_sig.position_pct
                    })

                elif acc_sig.side == 'short':
                    # === 숏 축적 (헤지용: 마진 개념으로 처리) ===
                    # 숏 진입 시 마진으로 alloc_amount 차감, 나중에 청산 시 수익/손실 반영
                    new_total_qty = short_accum_total_qty + qty
                    if new_total_qty > 0:
                        short_accum_avg_price = (short_accum_avg_price * short_accum_total_qty + fill_px * qty) / new_total_qty
                    short_accum_total_qty = new_total_qty
                    # 숏은 마진만 잠금 (전액이 아님)
                    margin_amount = alloc_amount * 0.1  # 10% 마진
                    cash -= margin_amount
                    short_accum_stop_price = acc_sig.stop_price
                    short_accum_tp_price = acc_sig.tp_price
                    short_accum_last_entry_i = i  # 진입 바 기록

                    short_accum_positions.append({
                        "time": t, "price": fill_px, "qty": qty,
                        "event": acc_sig.event, "phase": acc_sig.phase,
                        "margin": margin_amount
                    })
                    trades.append({
                        "time": t, "type": "ACCUM_SHORT", "side": "short",
                        "price": fill_px, "event": acc_sig.event,
                        "phase": acc_sig.phase, "position_pct": acc_sig.position_pct
                    })

        # === 롱 축적 포지션 청산 로직 (Wyckoff Phase 기반 스윙 전략 + 동적 파라미터) ===
        # 진입 바에서는 청산 검사 스킵 (다음 바부터 검사)
        if accum_total_qty > 0 and accum_avg_price > 0 and i > accum_last_entry_i:
            # Wyckoff 박스 레벨 및 Phase 정보 가져오기
            current_box_high = phases_df.loc[t, 'box_high']
            current_box_low = phases_df.loc[t, 'box_low']
            current_box_width = current_box_high - current_box_low
            current_phase = phases_df.loc[t, 'phase']
            current_direction = phases_df.loc[t, 'direction']

            # 동적 파라미터 가져오기
            dyn_params = dyn_params_df.loc[t]

            # 동적 TP/SL 비율 계산 (Wyckoff 박스 폭 기준)
            # 변동성/사이클/추세에 따라 동적으로 조정
            # tp1_ratio: 0.5 ~ 0.8 (저변동성: 박스 상단까지, 고변동성: 박스 중간)
            # tp2_ratio: 0.7 ~ 1.2 (저변동성: 박스 돌파, 고변동성: 박스 상단)
            # sl_ratio: 0.15 ~ 0.35 (저변동성: 좁은 SL, 고변동성: 넓은 SL)
            tp1_ratio = dyn_params['atr_mult_tp1']  # 동적 TP1 박스 비율
            tp2_ratio = dyn_params['atr_mult_tp2']  # 동적 TP2 박스 비율
            sl_ratio = dyn_params['atr_mult_stop']  # 동적 SL 박스 비율

            # 피보나치 되돌림 기반 동적 TP/SL
            # 평균 진입가 기준으로 박스 폭 × 동적 비율로 TP/SL 설정
            box_tp1 = accum_avg_price + (current_box_width * tp1_ratio)  # TP1: 동적 (변동성 기반)
            box_tp2 = accum_avg_price + (current_box_width * tp2_ratio)  # TP2: 동적 (변동성 기반)
            box_sl = accum_avg_price - (current_box_width * sl_ratio)  # SL: 동적 (변동성 기반)

            # TIME_EXIT: 최대 보유 기간 초과 시 강제 청산 (동적 보유 기간)
            hold_bars = i - accum_last_entry_i
            optimal_hold = int(dyn_params['optimal_hold_bars'])
            dynamic_max_hold = max(optimal_hold, cfg.max_hold_bars)  # 동적 또는 설정값 중 큰 값
            time_exit_triggered = hold_bars >= dynamic_max_hold

            # Phase 변화 감지: Phase A/B/C → D/E (Markup/Trend) 전환 시 청산
            # Phase D (Markup 시작) 이후 일정 기간 경과 시 청산
            # Phase C (Spring)은 여전히 축적 구간이므로 유지
            phase_exit_triggered = False  # Phase 기반 청산은 보수적으로 접근

            # 박스 변경 감지
            if i > 0:
                prev_box_high = phases_df.iloc[i-1]['box_high']
                prev_box_low = phases_df.iloc[i-1]['box_low']
                prev_box_width = prev_box_high - prev_box_low

                box_center_change = abs((current_box_high + current_box_low) / 2 - (prev_box_high + prev_box_low) / 2)
                box_size_change = abs(current_box_width - prev_box_width) / (prev_box_width + 1e-12)
                box_changed = (box_size_change > 0.5) or (box_center_change > prev_box_width * 0.5)
            else:
                box_changed = False

            # 1. TP2: 박스 폭 80% 도달 시 전체 청산 (고가가 TP2에 도달)
            if h >= box_tp2:
                exit_px = apply_costs(box_tp2, costs)
                pnl_pct = (exit_px - accum_avg_price) / accum_avg_price * 100
                cash += accum_total_qty * exit_px
                trades.append({
                    "time": t,
                    "type": "ACCUM_TP2",
                    "side": "long",
                    "price": exit_px,
                    "qty": accum_total_qty,
                    "avg_entry": accum_avg_price,
                    "pnl_pct": pnl_pct
                })
                accum_total_qty = 0.0
                accum_avg_price = 0.0
                accum_stop_price = np.nan
                accum_tp_price = np.nan
                accum_positions = []
            # 2. TP1: 박스 폭 50% 도달 시 부분 청산 (cfg.tp1_frac 비율)
            elif h >= box_tp1 and accum_total_qty > 0:
                exit_qty = accum_total_qty * cfg.tp1_frac  # 50% 청산
                exit_px = apply_costs(box_tp1, costs)
                pnl_pct = (exit_px - accum_avg_price) / accum_avg_price * 100
                cash += exit_qty * exit_px
                trades.append({
                    "time": t,
                    "type": "ACCUM_TP1",
                    "side": "long",
                    "price": exit_px,
                    "qty": exit_qty,
                    "avg_entry": accum_avg_price,
                    "pnl_pct": pnl_pct
                })
                accum_total_qty -= exit_qty
                # TP1 후 남은 포지션은 trailing stop으로 전환 (SL을 진입가로 이동)
                box_sl = accum_avg_price
            # 2. 손절: 박스 하단 돌파 (저가가 SL 이하)
            elif l <= box_sl:
                exit_px = apply_costs(box_sl, costs)
                pnl_pct = (exit_px - accum_avg_price) / accum_avg_price * 100
                cash += accum_total_qty * exit_px
                trades.append({
                    "time": t, "type": "ACCUM_SL", "side": "long",
                    "price": exit_px, "qty": accum_total_qty,
                    "avg_entry": accum_avg_price, "pnl_pct": pnl_pct
                })
                accum_total_qty = 0.0
                accum_avg_price = 0.0
                accum_stop_price = np.nan
                accum_tp_price = np.nan
                accum_positions = []
            # 3. Phase 전환 청산: Phase A/B → C/D/E (Markup 시작)
            elif phase_exit_triggered and hold_bars >= 8:  # 최소 8봉 이상 보유 후 Phase 전환 시
                exit_px = apply_costs(c, costs)
                pnl_pct = (exit_px - accum_avg_price) / accum_avg_price * 100
                cash += accum_total_qty * exit_px
                trades.append({
                    "time": t, "type": "PHASE_EXIT", "side": "long",
                    "price": exit_px, "qty": accum_total_qty,
                    "avg_entry": accum_avg_price, "pnl_pct": pnl_pct
                })
                accum_total_qty = 0.0
                accum_avg_price = 0.0
                accum_stop_price = np.nan
                accum_tp_price = np.nan
                accum_positions = []
            # 4. 박스 패턴 변경: 새로운 박스로 진입 시 기존 포지션 청산
            elif box_changed:
                exit_px = apply_costs(c, costs)
                pnl_pct = (exit_px - accum_avg_price) / accum_avg_price * 100
                cash += accum_total_qty * exit_px
                trades.append({
                    "time": t, "type": "BOX_CHANGE_EXIT", "side": "long",
                    "price": exit_px, "qty": accum_total_qty,
                    "avg_entry": accum_avg_price, "pnl_pct": pnl_pct
                })
                accum_total_qty = 0.0
                accum_avg_price = 0.0
                accum_stop_price = np.nan
                accum_tp_price = np.nan
                accum_positions = []
            # 5. TIME_EXIT: 최대 보유 기간 초과 시 강제 청산
            elif time_exit_triggered:
                exit_px = apply_costs(c, costs)
                pnl_pct = (exit_px - accum_avg_price) / accum_avg_price * 100
                cash += accum_total_qty * exit_px
                trades.append({
                    "time": t, "type": "TIME_EXIT", "side": "long",
                    "price": exit_px, "qty": accum_total_qty,
                    "avg_entry": accum_avg_price, "pnl_pct": pnl_pct
                })
                accum_total_qty = 0.0
                accum_avg_price = 0.0
                accum_stop_price = np.nan
                accum_tp_price = np.nan
                accum_positions = []

        # === 숏 축적 포지션 청산 로직 (Wyckoff Box 기반 스윙 전략 + 동적 파라미터) ===
        # 진입 바에서는 청산 검사 스킵 (다음 바부터 검사)
        if short_accum_total_qty > 0 and short_accum_avg_price > 0 and i > short_accum_last_entry_i:
            # Wyckoff 박스 레벨 가져오기 (phases_df)
            current_box_high = phases_df.loc[t, 'box_high']
            current_box_low = phases_df.loc[t, 'box_low']
            current_box_width = current_box_high - current_box_low

            # 동적 파라미터 가져오기
            dyn_params = dyn_params_df.loc[t]
            tp1_ratio = dyn_params['atr_mult_tp1']
            tp2_ratio = dyn_params['atr_mult_tp2']
            sl_ratio = dyn_params['atr_mult_stop']

            # 피보나치 되돌림 기반 동적 TP/SL (숏 포지션)
            box_tp1 = short_accum_avg_price - (current_box_width * tp1_ratio)  # TP1: 동적
            box_tp2 = short_accum_avg_price - (current_box_width * tp2_ratio)  # TP2: 동적
            box_sl = short_accum_avg_price + (current_box_width * sl_ratio)  # SL: 동적

            # TIME_EXIT: 최대 보유 기간 초과 시 강제 청산 (동적 보유 기간)
            hold_bars = i - short_accum_last_entry_i
            optimal_hold = int(dyn_params['optimal_hold_bars'])
            dynamic_max_hold = max(optimal_hold, cfg.max_hold_bars)
            time_exit_triggered = hold_bars >= dynamic_max_hold

            # 박스 변경 감지
            if i > 0:
                prev_box_high = phases_df.iloc[i-1]['box_high']
                prev_box_low = phases_df.iloc[i-1]['box_low']
                prev_box_width = prev_box_high - prev_box_low

                box_center_change = abs((current_box_high + current_box_low) / 2 - (prev_box_high + prev_box_low) / 2)
                box_size_change = abs(current_box_width - prev_box_width) / (prev_box_width + 1e-12)
                box_changed = (box_size_change > 0.5) or (box_center_change > prev_box_width * 0.5)
            else:
                box_changed = False

            total_margin = sum(p.get('margin', 0) for p in short_accum_positions)

            # 1. 숏 TP2: 박스 폭 80% 도달 시 전체 청산
            if l <= box_tp2:
                exit_px = apply_costs(box_tp2, costs)
                pnl_pct = (short_accum_avg_price - exit_px) / short_accum_avg_price * 100
                pnl_amount = short_accum_total_qty * (short_accum_avg_price - exit_px)
                cash += total_margin + pnl_amount
                trades.append({
                    "time": t, "type": "ACCUM_SHORT_TP2", "side": "short",
                    "price": exit_px, "qty": short_accum_total_qty,
                    "avg_entry": short_accum_avg_price, "pnl_pct": pnl_pct
                })
                short_accum_total_qty = 0.0
                short_accum_avg_price = 0.0
                short_accum_stop_price = np.nan
                short_accum_tp_price = np.nan
                short_accum_positions = []
            # 2. 숏 TP1: 박스 폭 50% 도달 시 부분 청산
            elif l <= box_tp1 and short_accum_total_qty > 0:
                exit_qty = short_accum_total_qty * cfg.tp1_frac
                exit_px = apply_costs(box_tp1, costs)
                pnl_pct = (short_accum_avg_price - exit_px) / short_accum_avg_price * 100
                pnl_amount = exit_qty * (short_accum_avg_price - exit_px)
                cash += (total_margin * cfg.tp1_frac) + pnl_amount
                trades.append({
                    "time": t, "type": "ACCUM_SHORT_TP1", "side": "short",
                    "price": exit_px, "qty": exit_qty,
                    "avg_entry": short_accum_avg_price, "pnl_pct": pnl_pct
                })
                short_accum_total_qty -= exit_qty
                # TP1 후 trailing stop
                box_sl = short_accum_avg_price
            # 2. 숏 손절: 박스 상단 돌파 (고가가 SL 이상)
            elif h >= box_sl:
                exit_px = apply_costs(box_sl, costs)
                pnl_pct = (short_accum_avg_price - exit_px) / short_accum_avg_price * 100
                raw_pnl = short_accum_total_qty * (short_accum_avg_price - exit_px)
                pnl_amount = max(raw_pnl, -total_margin)
                cash += total_margin + pnl_amount
                trades.append({
                    "time": t, "type": "ACCUM_SHORT_SL", "side": "short",
                    "price": exit_px, "qty": short_accum_total_qty,
                    "avg_entry": short_accum_avg_price, "pnl_pct": pnl_pct
                })
                short_accum_total_qty = 0.0
                short_accum_avg_price = 0.0
                short_accum_stop_price = np.nan
                short_accum_tp_price = np.nan
                short_accum_positions = []
            # 3. 박스 패턴 변경: 새로운 박스로 진입 시 기존 포지션 청산
            elif box_changed:
                exit_px = apply_costs(c, costs)
                pnl_pct = (short_accum_avg_price - exit_px) / short_accum_avg_price * 100
                raw_pnl = short_accum_total_qty * (short_accum_avg_price - exit_px)
                pnl_amount = max(raw_pnl, -total_margin)
                cash += total_margin + pnl_amount
                trades.append({
                    "time": t, "type": "BOX_CHANGE_EXIT", "side": "short",
                    "price": exit_px, "qty": short_accum_total_qty,
                    "avg_entry": short_accum_avg_price, "pnl_pct": pnl_pct
                })
                short_accum_total_qty = 0.0
                short_accum_avg_price = 0.0
                short_accum_stop_price = np.nan
                short_accum_tp_price = np.nan
                short_accum_positions = []
            # 4. TIME_EXIT: 최대 보유 기간 초과 시 강제 청산
            elif time_exit_triggered:
                exit_px = apply_costs(c, costs)
                pnl_pct = (short_accum_avg_price - exit_px) / short_accum_avg_price * 100
                raw_pnl = short_accum_total_qty * (short_accum_avg_price - exit_px)
                pnl_amount = max(raw_pnl, -total_margin)
                cash += total_margin + pnl_amount
                trades.append({
                    "time": t, "type": "TIME_EXIT", "side": "short",
                    "price": exit_px, "qty": short_accum_total_qty,
                    "avg_entry": short_accum_avg_price, "pnl_pct": pnl_pct
                })
                short_accum_total_qty = 0.0
                short_accum_avg_price = 0.0
                short_accum_stop_price = np.nan
                short_accum_tp_price = np.nan
                short_accum_positions = []

        # place new orders (activate next bar) if gate open
        if t in sig_by_time:
            gate_ok = bool(int(nav_gate.get(t, 1)))
            if gate_ok:
                for s in sig_by_time[t]:
                    pfill = approx_pfill_det(df, i, s.entry_price, theta.N_fill)
                    if pfill < theta.F_min:
                        continue
                    pending_orders.append({
                        "activate_i": i+1,
                        "expire_i": i+1+theta.N_fill,
                        "side": s.side,
                        "event": s.event,
                        "limit": float(s.entry_price),
                        "stop": float(s.stop_price),
                        "tp1": float(s.tp1),
                        "tp2": float(s.tp2),
                        "meta": s.meta or {}
                    })

        # fill pending if flat (only fill ONE order per bar)
        if pos_side == 0 and pending_orders:
            still = []
            filled_this_bar = False
            for od in pending_orders:
                if filled_this_bar:
                    still.append(od)
                    continue
                if i < od["activate_i"]:
                    still.append(od); continue
                if i > od["expire_i"]:
                    continue
                limit_px = od["limit"]
                if l <= limit_px <= h:
                    fill_px = apply_costs(limit_px, costs)
                    pos_side = +1 if od["side"] == "long" else -1

                    # Phase C 진입: 현금의 20%만 사용
                    phase_c_allocation = 0.20  # 20% 배분

                    # Phase C 롱 진입 시 축적 포지션 합산
                    if pos_side == 1 and accum_total_qty > 0:
                        # 20%만 추가 진입
                        alloc_cash = cash * phase_c_allocation
                        new_qty = alloc_cash / fill_px
                        total_qty = accum_total_qty + new_qty
                        # 평균 진입가 계산 (축적 + 신규)
                        entry_px = (accum_avg_price * accum_total_qty + fill_px * new_qty) / total_qty
                        pos_qty = total_qty
                        cash -= alloc_cash  # 20%만 차감
                        # 축적 포지션 리셋 (합산 완료)
                        accum_total_qty = 0.0
                        accum_avg_price = 0.0
                        accum_stop_price = np.nan
                        accum_tp_price = np.nan
                        accum_positions = []
                    elif pos_side == -1 and short_accum_total_qty > 0:
                        # Phase C 숏 진입 시 숏 축적 포지션 합산
                        alloc_cash = cash * phase_c_allocation
                        new_qty = alloc_cash / fill_px
                        total_qty = short_accum_total_qty + new_qty
                        # 평균 진입가 계산 (축적 + 신규)
                        entry_px = (short_accum_avg_price * short_accum_total_qty + fill_px * new_qty) / total_qty
                        pos_qty = total_qty
                        # 숏 마진 반환 후 새 마진 차감
                        total_margin = sum(p.get('margin', 0) for p in short_accum_positions)
                        cash += total_margin  # 기존 마진 반환
                        cash -= alloc_cash  # 20%만 차감
                        # 숏 축적 포지션 리셋 (합산 완료)
                        short_accum_total_qty = 0.0
                        short_accum_avg_price = 0.0
                        short_accum_stop_price = np.nan
                        short_accum_tp_price = np.nan
                        short_accum_positions = []
                    else:
                        # 축적 없이 Phase C 단독 진입: 20%만 사용
                        alloc_cash = cash * phase_c_allocation
                        pos_qty = alloc_cash / fill_px
                        entry_px = fill_px
                        cash -= alloc_cash  # 20%만 차감
                    stop_price = od["stop"]
                    tp1 = od["tp1"]
                    tp2 = od["tp2"]
                    entry_i = i
                    tp1_done = False
                    trades.append({"time": t, "type":"ENTRY", "side": od["side"], "price": fill_px, "event": od["event"], **od["meta"]})
                    filled_this_bar = True  # 한 바에 하나만 체결
                    # 나머지 주문은 모두 취소 (같은 시간의 중복 주문 방지)
                    continue
                else:
                    still.append(od)
            pending_orders = still

        # manage position
        if pos_side != 0:
            # stop
            if (pos_side == 1 and l <= stop_price) or (pos_side == -1 and h >= stop_price):
                exit_px = apply_costs(stop_price, costs)
                if pos_side == 1:  # 롱: 매도 (기존 cash에 추가)
                    cash = cash + pos_qty * exit_px
                else:  # 숏: 매수로 청산 (기존 cash에 추가)
                    pnl = pos_qty * (entry_px - exit_px)  # 숏이므로 가격이 오르면 손실
                    cash = cash + pos_qty * entry_px + pnl
                trades.append({"time": t, "type":"STOP", "side":"long" if pos_side==1 else "short", "price": exit_px})
                pos_qty = 0.0; pos_side = 0; entry_px=np.nan; stop_price=np.nan; tp1=np.nan; tp2=np.nan; entry_i=-1; tp1_done=False
            else:
                # tp1
                if not tp1_done:
                    if (pos_side == 1 and h >= tp1) or (pos_side == -1 and l <= tp1):
                        exit_px = apply_costs(tp1, costs)
                        frac = cfg.tp1_frac
                        qty_exit = pos_qty * frac
                        if pos_side == 1:  # 롱
                            cash = cash + qty_exit * exit_px
                        else:  # 숏
                            cash = cash + qty_exit * entry_px + qty_exit * (entry_px - exit_px)
                        pos_qty = pos_qty * (1-frac)
                        tp1_done = True
                        trades.append({"time": t, "type":"TP1", "side":"long" if pos_side==1 else "short", "price": exit_px})
                # tp2
                if cfg.use_tp2 and pos_side != 0:
                    if (pos_side == 1 and h >= tp2) or (pos_side == -1 and l <= tp2):
                        exit_px = apply_costs(tp2, costs)
                        if pos_side == 1:  # 롱
                            cash = cash + pos_qty * exit_px
                        else:  # 숏
                            cash = cash + pos_qty * entry_px + pos_qty * (entry_px - exit_px)
                        trades.append({"time": t, "type":"TP2", "side":"long" if pos_side==1 else "short", "price": exit_px})
                        pos_qty = 0.0; pos_side = 0; entry_px=np.nan; stop_price=np.nan; tp1=np.nan; tp2=np.nan; entry_i=-1; tp1_done=False
                # max hold
                if pos_side != 0 and cfg.max_hold_bars is not None:
                    if (i - entry_i) >= cfg.max_hold_bars:
                        exit_px = apply_costs(c, costs)
                        if pos_side == 1:  # 롱
                            cash = cash + pos_qty * exit_px
                        else:  # 숏
                            cash = cash + pos_qty * entry_px + pos_qty * (entry_px - exit_px)
                        trades.append({"time": t, "type":"TIME_EXIT", "side":"long" if pos_side==1 else "short", "price": exit_px})
                        pos_qty = 0.0; pos_side = 0; entry_px=np.nan; stop_price=np.nan; tp1=np.nan; tp2=np.nan; entry_i=-1; tp1_done=False

        # equity 계산: 포지션이 있으면 현재가로 평가
        if pos_side == 1:  # 롱
            equity_val = cash + pos_qty * c
        elif pos_side == -1:  # 숏
            equity_val = cash + pos_qty * entry_px + pos_qty * (entry_px - c)
        else:  # 포지션 없음
            equity_val = cash

        # 롱 축적 포지션 가치 추가
        if accum_total_qty > 0:
            equity_val += accum_total_qty * c

        # 숏 축적 포지션 가치 추가 (마진 + 미실현 손익, 손실은 마진으로 제한)
        if short_accum_total_qty > 0:
            total_margin = sum(p.get('margin', 0) for p in short_accum_positions)
            short_pnl = short_accum_total_qty * (short_accum_avg_price - c)
            # 손실이 마진보다 크면 마진만큼만 손실
            short_pnl_capped = max(short_pnl, -total_margin)
            equity_val += total_margin + short_pnl_capped

        equity_rows.append({
            "time": t,
            "equity": float(equity_val),
            "pos_qty": float(pos_qty),
            "pos_side": int(pos_side),
            "accum_qty": float(accum_total_qty),
            "accum_avg_px": float(accum_avg_price) if accum_avg_price > 0 else 0.0,
            "short_accum_qty": float(short_accum_total_qty),
            "short_accum_avg_px": float(short_accum_avg_price) if short_accum_avg_price > 0 else 0.0
        })

    equity_df = pd.DataFrame(equity_rows).set_index("time")
    equity_df["ret"] = equity_df["equity"].pct_change().fillna(0.0)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["time","type","side","price"])
    return equity_df, trades_df, signals_df, nav
