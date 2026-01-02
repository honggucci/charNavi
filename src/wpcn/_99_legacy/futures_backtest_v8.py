"""
선물 백테스트 엔진 V8 - 올바른 타임프레임 구조
Futures Backtest Engine V8 - Correct Timeframe Structure

핵심 구조 (사용자 피드백 반영):
- 4H: 추세 확인 (Wyckoff 구조만, EMA 제거)
- 1H: Wyckoff 구조 (스프링/업쓰러스트)
- 15M: Stoch RSI 필터 (과매수/과매도 구간)
- 5M: RSI 다이버전스 + ZigZag 예측 + 진입 타이밍 + Stoch RSI K/D 크로스 (타이밍)

변경사항 from V7:
- Stoch RSI: 15M에서 체크 (V7은 잘못됨)
- RSI 다이버전스: 5M에서 체크 (V7은 15M에서 체크 - 잘못됨)
- EMA 추세 필터: 제거 (사용자 피드백)
- ZigZag: 추가 (chart_navigator_v2에서)

FIXED (P0):
- 15M Stoch RSI 룩어헤드 버그 수정 (완성된 봉만 참조)
- RSI 다이버전스 룩어헤드 버그 수정 (lookback만큼 shift)
- except:pass 제거, 로깅 추가

FIXED (P1):
- MTF 정렬을 merge_asof로 변경 (루프 내 인덱스 찾기 제거)

FIXED (P2):
- 점수 기반 → 확률 기반 시스템 전환
- First-Passage Probability 모델 추가
- Wyckoff 페이즈별 drift 조정
- 캘리브레이션 추적 추가

FIXED (GPT 피드백 - P0급):
- (P0-6) 숏 부호 이중 반전 버그 수정 (is_long으로만 방향 처리)
- (P0-7) MC seed 전달로 재현성 확보 (bar_idx 기반)
- (P0-8) 확률 합 정규화 수정 (eps 클램프 후 재정규화)
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from wpcn.features.indicators import atr, rsi, stoch_rsi, detect_rsi_divergence
from wpcn.probability.barrier import (
    calculate_barrier_probability,
    BarrierProbability,
    estimate_momentum_from_price  # FIXED (P0-4): 가격 기반 모멘텀
)
from wpcn.probability.calibration import ProbabilityCalibrator
from wpcn.wyckoff.phases import detect_phase_signals, AccumulationSignal, DIRECTION_RATIOS


@dataclass
class V8Config:
    """V8 백테스트 설정"""
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
    warmup_bars: int = 500
    cooldown_bars: int = 8  # 2시간
    max_hold_bars: int = 48  # 12시간

    # Wyckoff (1시간봉)
    box_lookback: int = 50
    spring_threshold: float = 0.005

    # RSI/Stoch RSI
    rsi_period: int = 14
    stoch_rsi_period: int = 14
    div_lookback: int = 20

    # ZigZag
    zigzag_threshold: float = 0.02  # 2%

    # 비용
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage: float = 0.0001

    # P2: 확률 기반 설정
    use_probability_model: bool = True  # 확률 모델 사용 여부
    min_tp_probability: float = 0.55    # 최소 TP 확률 (55%)
    min_ev_r: float = 0.10              # 최소 EV(R-unit): p_tp*R - p_sl - costs >= 0.10
    min_expected_rr: float = 0.005      # 최소 기대 RR (%)

    # 피보나치 그리드 전략 (박스 피보나치 거미줄)
    use_phase_accumulation: bool = True  # 피보나치 그리드 활성화
    accumulation_pct: float = 0.03       # 각 레벨당 3% 배분
    accum_tp_pct: float = 0.015          # 축적 포지션 TP 1.5%
    accum_sl_pct: float = 0.010          # 축적 포지션 SL 1.0%

    # Navigation Gate (현물 엔진과 동일)
    use_navigation_gate: bool = False    # Navigation Gate 활성화 여부
    conf_min: float = 0.50               # 최소 확신도 (50%)
    edge_min: float = 0.60               # 최소 엣지 스코어 (60%)
    adx_trend: float = 15.0              # ADX 추세 임계값
    chaos_ret_atr: float = 3.0           # CHAOS 감지 임계값


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
    # P2: 확률 정보
    probability: Optional[BarrierProbability] = None
    trade_id: str = ""


class V8BacktestEngine:
    """
    V8 백테스트 엔진 - 올바른 타임프레임 구조

    핵심:
    - 15M: Stoch RSI 필터
    - 5M: RSI 다이버전스 + ZigZag + 진입
    - EMA 제거, Wyckoff만 사용
    """

    def __init__(self, config: V8Config = None):
        self.config = config or V8Config()
        self.position: Optional[Position] = None
        self.trades: List[Dict] = []
        self.signals: List[Dict] = []
        self.equity_history: List[Dict] = []
        # P2: 캘리브레이션 추적
        self.calibrator = ProbabilityCalibrator()
        self.trade_counter = 0

        # 피보나치 그리드 축적 포지션 (롱/숏 분리)
        self.long_accum_qty = 0.0
        self.long_accum_avg_price = 0.0
        self.long_accum_margin = 0.0
        self.long_accum_last_i = -10

        self.short_accum_qty = 0.0
        self.short_accum_avg_price = 0.0
        self.short_accum_margin = 0.0
        self.short_accum_last_i = -10

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
        self.calibrator.clear()
        self.trade_counter = 0

        # 피보나치 그리드 축적 포지션 초기화
        self.long_accum_qty = 0.0
        self.long_accum_avg_price = 0.0
        self.long_accum_margin = 0.0
        self.long_accum_last_i = -10
        self.short_accum_qty = 0.0
        self.short_accum_avg_price = 0.0
        self.short_accum_margin = 0.0
        self.short_accum_last_i = -10

        cash = initial_capital
        last_entry_bar = -cfg.cooldown_bars - 1

        # 리샘플링 (shift=True로 완성된 봉만 사용 - 룩어헤드 방지)
        df_15m = self._resample(df_5m, '15min')
        df_1h = self._resample(df_5m, '1h')
        df_4h = self._resample(df_5m, '4h')

        # 5분봉 지표 (메인 신호)
        atr_5m = atr(df_5m, 14)
        div_5m = detect_rsi_divergence(df_5m, rsi_len=cfg.rsi_period, lookback=cfg.div_lookback)
        df_5m['rsi'] = rsi(df_5m['close'], cfg.rsi_period)

        # ZigZag 계산 (5분봉)
        zigzag_pivots = self._calculate_zigzag(df_5m, cfg.zigzag_threshold)

        # 15분봉 Stoch RSI (필터용)
        stoch_k_15m, stoch_d_15m = stoch_rsi(df_15m['close'], cfg.stoch_rsi_period, cfg.stoch_rsi_period, 3, 3)
        df_15m['stoch_k'] = stoch_k_15m
        df_15m['stoch_d'] = stoch_d_15m

        # 5분봉 Stoch RSI (타이밍용)
        stoch_k_5m, stoch_d_5m = stoch_rsi(df_5m['close'], cfg.stoch_rsi_period, cfg.stoch_rsi_period, 3, 3)
        df_5m['stoch_k'] = stoch_k_5m
        df_5m['stoch_d'] = stoch_d_5m

        # === FIXED (P1): merge_asof로 MTF 지표 사전 병합 ===
        # 15분봉 지표를 5분봉에 병합 (완성된 봉만 - shift 적용)
        df_15m_shifted = df_15m[['stoch_k', 'stoch_d']].shift(1).dropna()
        df_5m = self._merge_asof_mtf(df_5m, df_15m_shifted, suffix='_15m')

        # 5분봉 다이버전스 병합
        df_5m = self._merge_asof_mtf(df_5m, div_5m[['bullish_div', 'bearish_div', 'div_strength']], suffix='_div')

        # === Navigation Gate 계산 (선택적) ===
        nav_gate = None
        if cfg.use_navigation_gate:
            from wpcn._03_common._01_core.types import BacktestConfig
            from wpcn._06_engine.navigation import compute_navigation

            # BacktestConfig 생성
            bt_cfg = BacktestConfig(
                initial_equity=1.0,
                max_hold_bars=cfg.max_hold_bars,
                conf_min=cfg.conf_min,
                edge_min=cfg.edge_min,
                adx_trend=cfg.adx_trend,
                chaos_ret_atr=cfg.chaos_ret_atr
            )

            # Theta 생성
            from wpcn.core.types import Theta
            theta_nav = Theta(
                pivot_lr=4, box_L=96, m_freeze=32, atr_len=14,
                x_atr=0.35, m_bw=0.10, N_reclaim=3, N_fill=5, F_min=0.70
            )

            # Navigation 계산
            nav_df = compute_navigation(df_5m, theta_nav, bt_cfg, mtf=['1h', '4h'])
            nav_gate = nav_df['trade_gate'].to_dict()
            print(f"  Navigation Gate: {sum(nav_gate.values())} / {len(nav_gate)} bars allowed")

        n = len(df_5m)
        idx = df_5m.index

        # === 피보나치 그리드 신호 생성 (Phase Accumulation) ===
        accum_by_time: Dict[Any, List[AccumulationSignal]] = {}
        if cfg.use_phase_accumulation:
            from wpcn.core.types import Theta
            theta = Theta(
                pivot_lr=4, box_L=96, m_freeze=32, atr_len=14,
                x_atr=0.35, m_bw=0.10, N_reclaim=3, N_fill=5, F_min=0.70
            )
            # detect_phase_signals에 rsi 컬럼이 없는 복사본 전달 (충돌 방지)
            df_5m_clean = df_5m.drop(columns=['rsi'], errors='ignore').copy()
            phases_df, spider_signals, safety_signals = detect_phase_signals(
                df_5m_clean, theta, accumulation_pct=cfg.accumulation_pct
            )
            all_accum_signals = spider_signals + safety_signals
            for sig in all_accum_signals:
                accum_by_time.setdefault(sig.t_signal, []).append(sig)
            print(f"  Fibonacci Grid: {len(spider_signals)} spider + {len(safety_signals)} safety = {len(all_accum_signals)} signals")

        print(f"Running V8 backtest: {n} bars (5m), {symbol}")
        print(f"  15M: Stoch RSI filter (merge_asof aligned)")
        print(f"  5M: RSI Divergence + ZigZag + Entry timing")

        for i in range(cfg.warmup_bars, n):
            t = idx[i]
            row = df_5m.iloc[i]
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            current_atr = atr_5m.iloc[i] if i < len(atr_5m) and not pd.isna(atr_5m.iloc[i]) else c * 0.01

            # === 1. 포지션 청산 체크 ===
            if self.position:
                exit_result = self._check_exit(self.position, h, l, c, i, t, cfg)
                if exit_result:
                    exit_price, exit_reason, pnl = exit_result
                    cash += self.position.margin + pnl

                    # P2 FIXED (F): 3-class 캘리브레이션 기록
                    if self.position.probability is not None:
                        self.calibrator.add_completed_record(
                            predicted_prob=self.position.probability.p_tp,
                            predicted_p_sl=self.position.probability.p_sl,
                            predicted_p_timeout=self.position.probability.p_timeout,
                            exit_reason=exit_reason,
                            timestamp=str(t),
                            trade_id=self.position.trade_id
                        )

                    trade_record = {
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
                    }

                    # P2 FIXED: 3-class 확률 정보 추가
                    if self.position.probability is not None:
                        trade_record['p_tp'] = self.position.probability.p_tp
                        trade_record['p_sl'] = self.position.probability.p_sl
                        trade_record['p_timeout'] = self.position.probability.p_timeout
                        trade_record['ev_r'] = self.position.probability.ev_r
                        trade_record['ev_pct'] = self.position.probability.ev_pct
                        trade_record['edge'] = self.position.probability.edge

                    self.trades.append(trade_record)
                    self.position = None

            # === 1.5 피보나치 그리드 축적 포지션 청산 체크 ===
            if cfg.use_phase_accumulation:
                # 롱 축적 포지션 청산
                if self.long_accum_qty > 0 and self.long_accum_avg_price > 0 and i > self.long_accum_last_i:
                    tp_target = self.long_accum_avg_price * (1 + cfg.accum_tp_pct)
                    sl_target = self.long_accum_avg_price * (1 - cfg.accum_sl_pct)

                    if h >= tp_target:  # TP
                        exit_px = tp_target * (1 - cfg.slippage)
                        pnl = self.long_accum_qty * (exit_px - self.long_accum_avg_price)
                        pnl -= self.long_accum_qty * exit_px * cfg.taker_fee
                        cash += self.long_accum_margin + pnl
                        self.trades.append({
                            'symbol': symbol, 'side': 'long', 'type': 'ACCUM_TP',
                            'entry_price': self.long_accum_avg_price, 'exit_price': exit_px,
                            'quantity': self.long_accum_qty, 'pnl': pnl,
                            'pnl_pct': pnl / self.long_accum_margin * 100 if self.long_accum_margin > 0 else 0,
                            'entry_time': t, 'exit_time': t, 'exit_reason': 'accum_tp',
                            'hold_bars': i - self.long_accum_last_i, 'score': 0, 'reasons': {'type': 'fibonacci_grid'}
                        })
                        self.long_accum_qty = 0.0
                        self.long_accum_avg_price = 0.0
                        self.long_accum_margin = 0.0

                    elif l <= sl_target:  # SL
                        exit_px = sl_target * (1 - cfg.slippage)
                        pnl = self.long_accum_qty * (exit_px - self.long_accum_avg_price)
                        pnl -= self.long_accum_qty * exit_px * cfg.taker_fee
                        cash += self.long_accum_margin + pnl
                        self.trades.append({
                            'symbol': symbol, 'side': 'long', 'type': 'ACCUM_SL',
                            'entry_price': self.long_accum_avg_price, 'exit_price': exit_px,
                            'quantity': self.long_accum_qty, 'pnl': pnl,
                            'pnl_pct': pnl / self.long_accum_margin * 100 if self.long_accum_margin > 0 else 0,
                            'entry_time': t, 'exit_time': t, 'exit_reason': 'accum_sl',
                            'hold_bars': i - self.long_accum_last_i, 'score': 0, 'reasons': {'type': 'fibonacci_grid'}
                        })
                        self.long_accum_qty = 0.0
                        self.long_accum_avg_price = 0.0
                        self.long_accum_margin = 0.0

                # 숏 축적 포지션 청산
                if self.short_accum_qty > 0 and self.short_accum_avg_price > 0 and i > self.short_accum_last_i:
                    tp_target = self.short_accum_avg_price * (1 - cfg.accum_tp_pct)
                    sl_target = self.short_accum_avg_price * (1 + cfg.accum_sl_pct)

                    if l <= tp_target:  # TP (숏은 가격 하락이 TP)
                        exit_px = tp_target * (1 + cfg.slippage)
                        pnl = self.short_accum_qty * (self.short_accum_avg_price - exit_px)
                        pnl -= self.short_accum_qty * exit_px * cfg.taker_fee
                        cash += self.short_accum_margin + pnl
                        self.trades.append({
                            'symbol': symbol, 'side': 'short', 'type': 'ACCUM_TP',
                            'entry_price': self.short_accum_avg_price, 'exit_price': exit_px,
                            'quantity': self.short_accum_qty, 'pnl': pnl,
                            'pnl_pct': pnl / self.short_accum_margin * 100 if self.short_accum_margin > 0 else 0,
                            'entry_time': t, 'exit_time': t, 'exit_reason': 'accum_tp',
                            'hold_bars': i - self.short_accum_last_i, 'score': 0, 'reasons': {'type': 'fibonacci_grid'}
                        })
                        self.short_accum_qty = 0.0
                        self.short_accum_avg_price = 0.0
                        self.short_accum_margin = 0.0

                    elif h >= sl_target:  # SL (숏은 가격 상승이 SL)
                        exit_px = sl_target * (1 + cfg.slippage)
                        pnl = self.short_accum_qty * (self.short_accum_avg_price - exit_px)
                        pnl -= self.short_accum_qty * exit_px * cfg.taker_fee
                        cash += self.short_accum_margin + pnl
                        self.trades.append({
                            'symbol': symbol, 'side': 'short', 'type': 'ACCUM_SL',
                            'entry_price': self.short_accum_avg_price, 'exit_price': exit_px,
                            'quantity': self.short_accum_qty, 'pnl': pnl,
                            'pnl_pct': pnl / self.short_accum_margin * 100 if self.short_accum_margin > 0 else 0,
                            'entry_time': t, 'exit_time': t, 'exit_reason': 'accum_sl',
                            'hold_bars': i - self.short_accum_last_i, 'score': 0, 'reasons': {'type': 'fibonacci_grid'}
                        })
                        self.short_accum_qty = 0.0
                        self.short_accum_avg_price = 0.0
                        self.short_accum_margin = 0.0

            # === 1.6 피보나치 그리드 신호 처리 (축적 진입) ===
            if cfg.use_phase_accumulation and t in accum_by_time and self.position is None:
                # Navigation Gate 체크
                if cfg.use_navigation_gate and nav_gate is not None:
                    if not nav_gate.get(t, False):
                        continue  # Navigation Gate 통과 실패

                for acc_sig in accum_by_time[t]:
                    alloc_amount = cash * acc_sig.position_pct * cfg.leverage
                    if alloc_amount < 10:  # 최소 금액
                        continue

                    fill_px = acc_sig.entry_price * (1 + cfg.slippage if acc_sig.side == 'long' else 1 - cfg.slippage)
                    qty = alloc_amount / fill_px
                    margin = alloc_amount / cfg.leverage
                    fee = alloc_amount * cfg.maker_fee

                    if acc_sig.side == 'long':
                        # 롱 축적
                        new_total_qty = self.long_accum_qty + qty
                        if new_total_qty > 0:
                            self.long_accum_avg_price = (
                                self.long_accum_avg_price * self.long_accum_qty + fill_px * qty
                            ) / new_total_qty
                        self.long_accum_qty = new_total_qty
                        self.long_accum_margin += margin
                        self.long_accum_last_i = i
                        cash -= margin + fee

                    else:
                        # 숏 축적
                        new_total_qty = self.short_accum_qty + qty
                        if new_total_qty > 0:
                            self.short_accum_avg_price = (
                                self.short_accum_avg_price * self.short_accum_qty + fill_px * qty
                            ) / new_total_qty
                        self.short_accum_qty = new_total_qty
                        self.short_accum_margin += margin
                        self.short_accum_last_i = i
                        cash -= margin + fee

                    self.trades.append({
                        'symbol': symbol, 'side': acc_sig.side, 'type': 'ACCUM_ENTRY',
                        'entry_price': fill_px, 'exit_price': 0.0,
                        'quantity': qty, 'pnl': 0.0, 'pnl_pct': 0.0,
                        'entry_time': t, 'exit_time': None, 'exit_reason': acc_sig.event,
                        'hold_bars': 0, 'score': 0,
                        'reasons': {'event': acc_sig.event, 'phase': acc_sig.phase, 'type': 'fibonacci_grid'}
                    })

            # === 2. 신호 분석 (5분봉 기준) ===
            # FIXED (P1): merge_asof로 병합된 df_5m 사용
            if self.position is None and i - last_entry_bar >= cfg.cooldown_bars:
                signal = self._analyze_signal(
                    df_5m, df_1h, df_4h,
                    zigzag_pivots,
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

                    # P2: 확률 모델 계산
                    barrier_prob = None
                    prob_valid = True

                    if cfg.use_probability_model:
                        # 신호에서 지표 추출
                        rsi_val = row.get('rsi', 50.0)
                        div_strength = row.get('div_strength_div', 0.0)
                        wyckoff_phase = signal.get('reasons', {}).get('1h_wyckoff', 'unknown').lower()

                        # FIXED (P0-4): 가격 기반 모멘텀 (score 순환 참조 제거)
                        # 최근 20봉의 종가에서 직접 모멘텀 계산
                        lookback = min(20, i)
                        recent_closes = df_5m['close'].iloc[max(0, i - lookback):i + 1].values
                        momentum = estimate_momentum_from_price(recent_closes, lookback=lookback)

                        # P2 FIXED (A): finite horizon (max_hold_bars) 전달
                        # FIXED (P0-5): use_mean_reversion=True (RSI 다이버전스 전략)
                        # FIXED (P0-6): 부호 뒤집기는 is_long으로만 처리 (이중 반전 방지)
                        # FIXED (P0-7): bar_idx 기반 seed로 MC 재현성 확보
                        mc_seed = hash(f"{symbol}_{i}") % (2**31)  # 재현 가능한 seed
                        barrier_prob = calculate_barrier_probability(
                            current_price=entry_price,
                            tp_price=tp_price,
                            sl_price=sl_price,
                            atr=current_atr,
                            max_hold_bars=cfg.max_hold_bars,  # FIXED: 시간 제한 반영
                            momentum=momentum,  # FIXED: 원본 그대로 (is_long으로 방향 처리)
                            rsi=rsi_val,
                            div_strength=div_strength,  # FIXED: 원본 그대로
                            wyckoff_phase=wyckoff_phase,
                            phase_confidence=0.6,
                            is_long=(direction == 'long'),
                            entry_fee_pct=cfg.maker_fee,
                            exit_fee_pct=cfg.taker_fee,
                            slippage_pct=cfg.slippage,
                            use_monte_carlo=True,
                            n_simulations=2000,
                            use_mean_reversion=True,  # FIXED (P0-5): 평균회귀 전략
                            seed=mc_seed  # FIXED (P0-7): 재현성 seed
                        )

                        # P2 FIXED (E): EV 기반 조건 체크
                        prob_valid = (
                            barrier_prob.p_tp >= cfg.min_tp_probability and
                            barrier_prob.ev_r >= cfg.min_ev_r and  # FIXED: 네이밍 통일
                            barrier_prob.ev_pct >= cfg.min_expected_rr
                        )

                        signal['p_tp'] = barrier_prob.p_tp
                        signal['p_sl'] = barrier_prob.p_sl
                        signal['p_timeout'] = barrier_prob.p_timeout
                        signal['ev_r'] = barrier_prob.ev_r
                        signal['ev_pct'] = barrier_prob.ev_pct

                    if rr >= cfg.min_rr_ratio and prob_valid:
                        position_value = cash * cfg.position_pct * cfg.leverage
                        quantity = position_value / entry_price
                        margin = position_value / cfg.leverage
                        fee = position_value * cfg.maker_fee

                        # P2: 트레이드 ID 생성
                        self.trade_counter += 1
                        trade_id = f"{symbol}_{i}_{self.trade_counter}"

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
                            reasons=signal['reasons'],
                            probability=barrier_prob,
                            trade_id=trade_id
                        )

                        cash -= margin + fee
                        last_entry_bar = i

            # === Equity 기록 ===
            if i % 60 == 0:  # 5시간마다 기록
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

    def _merge_asof_mtf(
        self,
        df_base: pd.DataFrame,
        df_higher: pd.DataFrame,
        suffix: str = ""
    ) -> pd.DataFrame:
        """
        MTF 지표를 merge_asof로 병합 (룩어헤드 방지)

        Args:
            df_base: 기준 DataFrame (5분봉)
            df_higher: 상위 TF 지표 DataFrame
            suffix: 컬럼 접미사

        Returns:
            병합된 DataFrame
        """
        # 인덱스를 컬럼으로 변환
        df_base_reset = df_base.reset_index()
        df_higher_reset = df_higher.reset_index()

        # 인덱스 컬럼명 통일
        base_time_col = df_base_reset.columns[0]
        higher_time_col = df_higher_reset.columns[0]
        df_higher_reset = df_higher_reset.rename(columns={higher_time_col: base_time_col})

        # 접미사 추가
        if suffix:
            rename_dict = {
                col: f"{col}{suffix}"
                for col in df_higher_reset.columns
                if col != base_time_col
            }
            df_higher_reset = df_higher_reset.rename(columns=rename_dict)

        # merge_asof: backward로 과거 데이터만 참조
        merged = pd.merge_asof(
            df_base_reset,
            df_higher_reset,
            on=base_time_col,
            direction='backward'
        )

        # 인덱스 복원
        merged = merged.set_index(base_time_col)

        return merged

    def _calculate_zigzag(self, df: pd.DataFrame, threshold: float) -> List[Dict]:
        """ZigZag 피벗 계산"""
        pivots = []

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        if len(close) < 10:
            return pivots

        last_pivot_type = None
        last_pivot_price = close[0]
        last_pivot_idx = 0

        for i in range(1, len(close)):
            current_high = high[i]
            current_low = low[i]

            if last_pivot_type is None or last_pivot_type == 'low':
                change = (current_high - last_pivot_price) / last_pivot_price
                if change >= threshold:
                    if pivots and pivots[-1]['type'] == 'high':
                        if current_high > pivots[-1]['price']:
                            pivots[-1] = {'idx': i, 'price': current_high, 'type': 'high'}
                    else:
                        pivots.append({'idx': i, 'price': current_high, 'type': 'high'})
                    last_pivot_type = 'high'
                    last_pivot_price = current_high
                    last_pivot_idx = i

            if last_pivot_type is None or last_pivot_type == 'high':
                change = (last_pivot_price - current_low) / last_pivot_price
                if change >= threshold:
                    if pivots and pivots[-1]['type'] == 'low':
                        if current_low < pivots[-1]['price']:
                            pivots[-1] = {'idx': i, 'price': current_low, 'type': 'low'}
                    else:
                        pivots.append({'idx': i, 'price': current_low, 'type': 'low'})
                    last_pivot_type = 'low'
                    last_pivot_price = current_low
                    last_pivot_idx = i

        return pivots

    def _predict_zigzag(
        self,
        pivots: List[Dict],
        current_idx: int,
        current_price: float,
        wyckoff_phase: str,
        has_div: bool,
        div_type: str
    ) -> Dict:
        """ZigZag 기반 다음 스윙 예측"""
        if len(pivots) < 2:
            return {'direction': 'none', 'probability': 0.5, 'target': current_price}

        # 현재 인덱스보다 이전의 피벗만 사용
        valid_pivots = [p for p in pivots if p['idx'] < current_idx]
        if len(valid_pivots) < 2:
            return {'direction': 'none', 'probability': 0.5, 'target': current_price}

        last_pivot = valid_pivots[-1]
        prev_pivot = valid_pivots[-2]

        # 스윙 범위
        swing_range = abs(last_pivot['price'] - prev_pivot['price'])

        # 기본 방향 (마지막 피벗의 반대)
        if last_pivot['type'] == 'high':
            base_direction = 'down'
            base_target = last_pivot['price'] - swing_range * 0.618
        else:
            base_direction = 'up'
            base_target = last_pivot['price'] + swing_range * 0.618

        # 확률 조정
        prob = 0.5

        # Wyckoff 구조에 따른 조정
        if wyckoff_phase in ['accumulation', 're_accumulation']:
            if base_direction == 'up':
                prob += 0.15
            else:
                prob -= 0.1
        elif wyckoff_phase in ['distribution', 're_distribution']:
            if base_direction == 'down':
                prob += 0.15
            else:
                prob -= 0.1
        elif wyckoff_phase == 'markup':
            if base_direction == 'up':
                prob += 0.1
        elif wyckoff_phase == 'markdown':
            if base_direction == 'down':
                prob += 0.1

        # 다이버전스에 따른 조정
        if has_div:
            if div_type == 'bullish' and base_direction == 'up':
                prob += 0.2
            elif div_type == 'bearish' and base_direction == 'down':
                prob += 0.2

        return {
            'direction': base_direction,
            'probability': min(0.85, max(0.2, prob)),
            'target': base_target
        }

    def _analyze_signal(
        self,
        df_5m: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        zigzag_pivots: List[Dict],
        bar_idx: int,
        timestamp: datetime,
        atr_val: float,
        current_price: float
    ) -> Optional[Dict]:
        """
        신호 분석 - V8 구조

        FIXED (P1): merge_asof로 병합된 컬럼 사용
        - df_5m에 'stoch_k_15m', 'stoch_d_15m' 컬럼 포함
        - df_5m에 'bullish_div_div', 'bearish_div_div', 'div_strength_div' 컬럼 포함
        """
        cfg = self.config

        long_score = 0.0
        short_score = 0.0
        reasons = {}
        tf_alignment = 0

        # 현재 행
        row = df_5m.iloc[bar_idx]

        # 타임스탬프 (Wyckoff 분석용)
        t_1h_prev = timestamp.floor('1h') - timedelta(hours=1)
        t_4h_prev = timestamp.floor('4h') - timedelta(hours=4)

        # === 4H Wyckoff (EMA 제거) ===
        wyckoff_4h_phase = 'unknown'
        if t_4h_prev in df_4h.index:
            wyckoff_4h = self._analyze_wyckoff(df_4h, t_4h_prev, cfg)
            wyckoff_4h_phase = wyckoff_4h['phase']
            reasons['4h_wyckoff'] = wyckoff_4h_phase.upper()

            if wyckoff_4h_phase in ['accumulation', 're_accumulation', 'markup']:
                long_score += 1.5
                tf_alignment += 1
            elif wyckoff_4h_phase in ['distribution', 're_distribution', 'markdown']:
                short_score += 1.5
                tf_alignment += 1

        # === 1H Wyckoff ===
        wyckoff_1h_phase = 'unknown'
        spring_1h = False
        upthrust_1h = False

        if t_1h_prev in df_1h.index:
            wyckoff_1h = self._analyze_wyckoff(df_1h, t_1h_prev, cfg)
            wyckoff_1h_phase = wyckoff_1h['phase']
            spring_1h = wyckoff_1h.get('spring', False)
            upthrust_1h = wyckoff_1h.get('upthrust', False)

            reasons['1h_wyckoff'] = wyckoff_1h_phase.upper()

            if wyckoff_1h_phase in ['accumulation', 're_accumulation']:
                long_score += 2.0
                tf_alignment += 1
            elif wyckoff_1h_phase in ['distribution', 're_distribution']:
                short_score += 2.0
                tf_alignment += 1
            elif wyckoff_1h_phase == 'markup':
                long_score += 1.0
            elif wyckoff_1h_phase == 'markdown':
                short_score += 1.0

            if spring_1h:
                long_score += 2.0
                reasons['1h_event'] = 'SPRING'
            elif upthrust_1h:
                short_score += 2.0
                reasons['1h_event'] = 'UPTHRUST'

        # === 15M Stoch RSI 필터 (핵심!) ===
        # FIXED (P1): merge_asof로 병합된 컬럼에서 직접 읽음 (룩어헤드 방지)
        stoch_15m_oversold = False
        stoch_15m_overbought = False

        k_15m = row.get('stoch_k_15m', np.nan)
        d_15m = row.get('stoch_d_15m', np.nan)

        if not pd.isna(k_15m) and not pd.isna(d_15m):
            # stoch_rsi 함수는 0-1 범위 반환 (0-100이 아님)
            # 임계값: 30/70으로 완화 (더 많은 신호)
            if k_15m < 0.30:  # 과매도 (30%)
                stoch_15m_oversold = True
                long_score += 2.0
                reasons['15m_stoch'] = f'OVERSOLD ({k_15m*100:.1f})'
                tf_alignment += 1
            elif k_15m > 0.70:  # 과매수 (70%)
                stoch_15m_overbought = True
                short_score += 2.0
                reasons['15m_stoch'] = f'OVERBOUGHT ({k_15m*100:.1f})'
                tf_alignment += 1
            else:
                reasons['15m_stoch'] = f'NEUTRAL ({k_15m*100:.1f})'

        # === 5M RSI 다이버전스 (핵심!) ===
        # FIXED (P1): merge_asof로 병합된 컬럼에서 직접 읽음
        has_bullish_div = False
        has_bearish_div = False
        div_strength_val = 0.0

        bullish_div = row.get('bullish_div_div', False)
        bearish_div = row.get('bearish_div_div', False)
        div_strength_val = row.get('div_strength_div', 0.0)

        if bullish_div:
            has_bullish_div = True

            # 15M Stoch RSI가 과매도일 때만 롱 진입
            if stoch_15m_oversold:
                long_score += 2.5 * min(div_strength_val, 1.0)
                reasons['5m_div'] = f'BULLISH ({div_strength_val:.2f})'
                tf_alignment += 1
            else:
                long_score += 0.5  # 필터 없으면 약한 점수
                reasons['5m_div'] = f'BULLISH_UNFILTERED ({div_strength_val:.2f})'

        elif bearish_div:
            has_bearish_div = True

            # 15M Stoch RSI가 과매수일 때만 숏 진입
            if stoch_15m_overbought:
                short_score += 2.5 * min(div_strength_val, 1.0)
                reasons['5m_div'] = f'BEARISH ({div_strength_val:.2f})'
                tf_alignment += 1
            else:
                short_score += 0.5
                reasons['5m_div'] = f'BEARISH_UNFILTERED ({div_strength_val:.2f})'

        # === ZigZag 예측 ===
        div_type = 'bullish' if has_bullish_div else ('bearish' if has_bearish_div else 'none')
        zigzag_pred = self._predict_zigzag(
            zigzag_pivots, bar_idx, current_price,
            wyckoff_1h_phase, has_bullish_div or has_bearish_div, div_type
        )

        if zigzag_pred['probability'] > 0.55:
            if zigzag_pred['direction'] == 'up':
                long_score += 1.5 * zigzag_pred['probability']
                reasons['zigzag'] = f"UP ({zigzag_pred['probability']:.0%})"
            elif zigzag_pred['direction'] == 'down':
                short_score += 1.5 * zigzag_pred['probability']
                reasons['zigzag'] = f"DOWN ({zigzag_pred['probability']:.0%})"

        # === 5M Stoch RSI K/D 크로스 (타이밍) ===
        # FIXED (P1): df_5m에 이미 stoch_k, stoch_d 컬럼이 있음
        k_5m = row.get('stoch_k', np.nan)
        d_5m = row.get('stoch_d', np.nan)

        if bar_idx > 0 and not pd.isna(k_5m) and not pd.isna(d_5m):
            prev_row = df_5m.iloc[bar_idx - 1]
            k_prev = prev_row.get('stoch_k', np.nan)
            d_prev = prev_row.get('stoch_d', np.nan)

            if not pd.isna(k_prev) and not pd.isna(d_prev):
                # stoch_rsi는 0-1 범위
                # 골든 크로스 (과매도에서)
                if k_prev < d_prev and k_5m > d_5m and k_5m < 0.30:
                    long_score += 1.0
                    reasons['5m_cross'] = 'GOLDEN'
                # 데드 크로스 (과매수에서)
                elif k_prev > d_prev and k_5m < d_5m and k_5m > 0.70:
                    short_score += 1.0
                    reasons['5m_cross'] = 'DEATH'

        # === 반전 캔들 ===
        candle = self._check_reversal_candle(df_5m, bar_idx)
        if candle == 'bullish':
            long_score += 1.0
            reasons['5m_candle'] = 'HAMMER'
        elif candle == 'bearish':
            short_score += 1.0
            reasons['5m_candle'] = 'SHOOTING_STAR'

        # === 신호 결정 ===
        action = 'none'
        score = 0.0

        # 조건: Stoch RSI 필터만 (30/70 완화) + tf_alignment
        # 다이버전스는 보조 점수로만 사용 (AND 아님)
        long_valid = (
            stoch_15m_oversold and  # 15M 과매도 (30%)
            tf_alignment >= cfg.min_tf_alignment
        )

        short_valid = (
            stoch_15m_overbought and  # 15M 과매수 (70%)
            tf_alignment >= cfg.min_tf_alignment
        )

        if long_score >= cfg.min_score and long_score > short_score + 1.0 and long_valid:
            action = 'long'
            score = long_score
        elif short_score >= cfg.min_score and short_score > long_score + 1.0 and short_valid:
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

    def _analyze_wyckoff(self, df: pd.DataFrame, timestamp, cfg: V8Config) -> Dict:
        """Wyckoff 분석"""
        try:
            idx = df.index.get_loc(timestamp)
        except KeyError:
            return {'phase': 'unknown'}

        if idx < cfg.box_lookback:
            return {'phase': 'unknown'}

        window = df.iloc[idx - cfg.box_lookback:idx + 1]

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

        # 페이즈 결정 (EMA 없이)
        phase = 'ranging'

        if spring:
            phase = 'accumulation'
        elif price_position < 0.3 and higher_lows and accumulation_volume:
            phase = 'accumulation'
        elif upthrust:
            phase = 'distribution'
        elif price_position > 0.7 and lower_highs and distribution_volume:
            phase = 'distribution'
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
        cfg: V8Config
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
        """현재 자산 (메인 포지션 + 축적 포지션 포함)"""
        equity = cash

        # 메인 포지션
        if self.position:
            if self.position.side == 'long':
                unrealized = (price - self.position.entry_price) * self.position.quantity
            else:
                unrealized = (self.position.entry_price - price) * self.position.quantity
            equity += self.position.margin + unrealized

        # 롱 축적 포지션
        if self.long_accum_qty > 0:
            long_unrealized = (price - self.long_accum_avg_price) * self.long_accum_qty
            equity += self.long_accum_margin + long_unrealized

        # 숏 축적 포지션
        if self.short_accum_qty > 0:
            short_unrealized = (self.short_accum_avg_price - price) * self.short_accum_qty
            equity += self.short_accum_margin + short_unrealized

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
            # FIXED: 청산된 거래만 통계에 포함 (ENTRY 제외)
            closed_trades = trades_df[~trades_df['type'].str.contains('ENTRY', case=False, na=False)]
            if len(closed_trades) > 0:
                wins = closed_trades[closed_trades['pnl'] > 0]
                losses = closed_trades[closed_trades['pnl'] < 0]
                win_rate = len(wins) / len(closed_trades) * 100
                total_profit = wins['pnl'].sum() if len(wins) > 0 else 0
                total_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                avg_pnl = closed_trades['pnl'].mean()
            else:
                win_rate = 0
                profit_factor = 0
                avg_pnl = 0
            long_trades = trades_df[trades_df['side'] == 'long']
            short_trades = trades_df[trades_df['side'] == 'short']
        else:
            win_rate = 0
            profit_factor = 0
            avg_pnl = 0
            long_trades = pd.DataFrame()
            short_trades = pd.DataFrame()

        # FIXED: 청산된 거래 수만 카운트
        closed_count = 0
        closed_long = 0
        closed_short = 0
        if len(trades_df) > 0:
            closed_trades_for_count = trades_df[~trades_df['type'].str.contains('ENTRY', case=False, na=False)]
            closed_count = len(closed_trades_for_count)
            closed_long = len(closed_trades_for_count[closed_trades_for_count['side'] == 'long'])
            closed_short = len(closed_trades_for_count[closed_trades_for_count['side'] == 'short'])

        stats = {
            'symbol': symbol,
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': closed_count,
            'long_trades': closed_long,
            'short_trades': closed_short,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_pnl': avg_pnl,
            'total_signals': len(signals_df),
            'total_entries': len(trades_df) - closed_count  # 진입 횟수 (참고용)
        }

        if len(trades_df) > 0 and 'exit_reason' in trades_df.columns:
            # accum_tp, accum_sl도 포함
            stats['tp_exits'] = len(trades_df[trades_df['exit_reason'].str.contains('tp', case=False, na=False)])
            stats['sl_exits'] = len(trades_df[trades_df['exit_reason'].str.contains('sl', case=False, na=False)])
            stats['timeout_exits'] = len(trades_df[trades_df['exit_reason'] == 'timeout'])
            stats['liquidations'] = len(trades_df[trades_df['exit_reason'] == 'liquidation'])

        # P2 FIXED (F): 3-class 캘리브레이션 결과 추가
        calibration_result = self.calibrator.analyze()
        if calibration_result:
            stats['brier_score'] = calibration_result.brier_score
            stats['brier_score_3class'] = calibration_result.brier_score_3class
            stats['calibration_error'] = calibration_result.calibration_error
            stats['overconfidence'] = calibration_result.overconfidence
            stats['correction_factor'] = self.calibrator.get_correction_factor()
            stats['timeout_rate'] = self.calibrator.get_timeout_rate()
            stats['outcome_distribution'] = calibration_result.outcome_distribution

        # P2 FIXED: 확률 모델 통계 (EV 기반)
        if len(trades_df) > 0 and 'p_tp' in trades_df.columns:
            stats['avg_predicted_p_tp'] = trades_df['p_tp'].mean()
            stats['avg_predicted_p_sl'] = trades_df['p_sl'].mean() if 'p_sl' in trades_df.columns else 0
            stats['avg_predicted_p_timeout'] = trades_df['p_timeout'].mean() if 'p_timeout' in trades_df.columns else 0
            stats['avg_ev_r'] = trades_df['ev_r'].mean() if 'ev_r' in trades_df.columns else 0
            stats['avg_ev_pct'] = trades_df['ev_pct'].mean() if 'ev_pct' in trades_df.columns else 0

        return {
            'equity_df': equity_df,
            'trades_df': trades_df,
            'signals_df': signals_df,
            'stats': stats,
            'calibration': calibration_result.to_dict() if calibration_result else None
        }
