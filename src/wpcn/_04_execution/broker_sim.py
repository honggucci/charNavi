from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

from wpcn._03_common._01_core.types import Theta, BacktestCosts, BacktestConfig
from wpcn._04_execution.cost import apply_costs
from wpcn._03_common._03_wyckoff.events import detect_spring_utad, detect_all_signals
from wpcn._03_common._03_wyckoff.phases import detect_phase_signals, AccumulationSignal
from wpcn._06_engine.navigation import compute_navigation

def approx_pfill_det(df: pd.DataFrame, t_idx: int, entry_price: float, N_fill: int) -> float:
    end = min(len(df), t_idx + 1 + N_fill)
    window = df.iloc[t_idx+1:end]
    if window.empty:
        return 0.0
    hit = ((window["low"] <= entry_price) & (entry_price <= window["high"])).any()
    return 1.0 if hit else 0.0

def simulate(df: pd.DataFrame, theta: Theta, costs: BacktestCosts, cfg: BacktestConfig, mtf: List[str] | None = None, scalping_mode: bool = False, use_phase_accumulation: bool = True, spot_mode: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    백테스트 시뮬레이션

    Args:
        spot_mode: True면 현물 (숏 비활성화), False면 선물 (숏 가능)
    """
    df = df.sort_index()
    mtf = mtf or []

    # navigation (page probs + Unknown gate)
    nav = compute_navigation(df, theta, cfg, mtf)

    # Use scalping signals if scalping_mode is enabled
    signals = detect_all_signals(df, theta, scalping_mode=scalping_mode)

    # Phase A/B accumulation signals (거미줄 + 안전마진)
    if use_phase_accumulation:
        phases_df, spider_signals, safety_signals = detect_phase_signals(
            df, theta, accumulation_pct=0.03  # 3%씩 매수 (자산 배분 강화)
        )
        accum_signals = spider_signals + safety_signals
        # 현물 모드: 숏 신호 제거
        if spot_mode:
            accum_signals = [s for s in accum_signals if s.side == 'long']
    else:
        phases_df = None
        accum_signals = []

    # 현물 모드: 메인 신호에서도 숏 제거
    if spot_mode:
        signals = [s for s in signals if s.side == 'long']
    signals_df = pd.DataFrame([{
        "time": s.t_signal, "side": s.side, "event": s.event,
        "entry": s.entry_price, "stop": s.stop_price, "tp1": s.tp1, "tp2": s.tp2,
        **{f"meta_{k}": v for k,v in (s.meta or {}).items()}
    } for s in signals]).sort_values("time") if signals else pd.DataFrame(columns=["time"])

    cash = float(cfg.initial_equity)
    pos_qty = 0.0
    pos_side = 0  # +1 long, -1 short
    entry_px = np.nan  # 진입 가격 저장
    stop_price = np.nan
    tp1 = np.nan
    tp2 = np.nan
    entry_i = -1
    tp1_done = False

    # Phase A/B 롱 축적 포지션 (별도 관리)
    accum_positions: List[Dict[str, Any]] = []  # 축적된 포지션들
    accum_total_qty = 0.0  # 축적된 총 수량
    accum_avg_price = 0.0  # 평균 매수가
    accum_stop_price = np.nan  # 축적 포지션 손절가
    accum_tp_price = np.nan  # 축적 포지션 익절가
    accum_last_entry_i = -10  # 마지막 진입 바 인덱스

    # Phase A/B 숏 축적 포지션 (별도 관리)
    short_accum_positions: List[Dict[str, Any]] = []
    short_accum_total_qty = 0.0
    short_accum_avg_price = 0.0
    short_accum_stop_price = np.nan
    short_accum_tp_price = np.nan
    short_accum_last_entry_i = -10  # 마지막 진입 바 인덱스

    pending_orders: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    sig_by_time: Dict[Any, List[Any]] = {}
    for s in signals:
        sig_by_time.setdefault(s.t_signal, []).append(s)

    # 축적 신호를 시간별로 인덱싱
    accum_by_time: Dict[Any, List[AccumulationSignal]] = {}
    for a in accum_signals:
        accum_by_time.setdefault(a.t_signal, []).append(a)

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

        # === Phase A/B 축적 신호 처리 (양방향) ===
        if t in accum_by_time and pos_side == 0:  # 메인 포지션 없을 때만 축적
            for acc_sig in accum_by_time[t]:
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

        # === 롱 축적 포지션 익절/손절 처리 (평균 진입가 기준 동적 계산) ===
        # 진입 바에서는 익절/손절 검사 스킵 (다음 바부터 검사)
        if accum_total_qty > 0 and accum_avg_price > 0 and i > accum_last_entry_i:
            # 15분봉 스윙 전략: 비용(0.4%) 대비 충분한 마진 확보
            # TP 1.5% / SL 1.0% = 손익비 1.5:1 (비용 고려 시 실질 1.1% / 0.6%)
            dynamic_tp = accum_avg_price * 1.015  # +1.5% 익절
            dynamic_sl = accum_avg_price * 0.990  # -1.0% 손절

            # 익절 검사 (고가가 TP에 도달)
            if h >= dynamic_tp:
                exit_px = apply_costs(dynamic_tp, costs)
                pnl_pct = (exit_px - accum_avg_price) / accum_avg_price * 100
                cash += accum_total_qty * exit_px
                trades.append({
                    "time": t,
                    "type": "ACCUM_TP",
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
            # 손절 검사 (저가가 SL에 도달)
            elif l <= dynamic_sl:
                exit_px = apply_costs(dynamic_sl, costs)
                pnl_pct = (exit_px - accum_avg_price) / accum_avg_price * 100
                cash += accum_total_qty * exit_px
                trades.append({
                    "time": t, "type": "ACCUM_LONG_STOP", "side": "long",
                    "price": exit_px, "qty": accum_total_qty,
                    "avg_entry": accum_avg_price, "pnl_pct": pnl_pct
                })
                accum_total_qty = 0.0
                accum_avg_price = 0.0
                accum_stop_price = np.nan
                accum_tp_price = np.nan
                accum_positions = []

        # === 숏 축적 포지션 익절/손절 처리 (평균 진입가 기준 동적 계산) ===
        # 진입 바에서는 익절/손절 검사 스킵 (다음 바부터 검사)
        if short_accum_total_qty > 0 and short_accum_avg_price > 0 and i > short_accum_last_entry_i:
            # 15분봉 스윙 전략: 비용(0.4%) 대비 충분한 마진 확보
            # TP 1.5% / SL 1.0% = 손익비 1.5:1
            dynamic_tp = short_accum_avg_price * 0.985  # -1.5% 익절 (숏)
            dynamic_sl = short_accum_avg_price * 1.010  # +1.0% 손절 (숏)

            total_margin = sum(p.get('margin', 0) for p in short_accum_positions)

            # 숏 익절 검사 (저가가 TP에 도달)
            if l <= dynamic_tp:
                exit_px = apply_costs(dynamic_tp, costs)
                pnl_pct = (short_accum_avg_price - exit_px) / short_accum_avg_price * 100
                pnl_amount = short_accum_total_qty * (short_accum_avg_price - exit_px)
                cash += total_margin + pnl_amount
                trades.append({
                    "time": t, "type": "ACCUM_SHORT_TP", "side": "short",
                    "price": exit_px, "qty": short_accum_total_qty,
                    "avg_entry": short_accum_avg_price, "pnl_pct": pnl_pct
                })
                short_accum_total_qty = 0.0
                short_accum_avg_price = 0.0
                short_accum_stop_price = np.nan
                short_accum_tp_price = np.nan
                short_accum_positions = []
            # 숏 손절 검사 (고가가 SL에 도달)
            elif h >= dynamic_sl:
                exit_px = apply_costs(dynamic_sl, costs)
                pnl_pct = (short_accum_avg_price - exit_px) / short_accum_avg_price * 100
                raw_pnl = short_accum_total_qty * (short_accum_avg_price - exit_px)
                pnl_amount = max(raw_pnl, -total_margin)
                cash += total_margin + pnl_amount
                trades.append({
                    "time": t, "type": "ACCUM_SHORT_STOP", "side": "short",
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
