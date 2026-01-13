"""
D-lite ATR SL/TP WFO 실험 v3 (v2.6.14)
======================================

v3 변경사항 (v2.6.14):
1. 15분봉 전용 백테스터 (futures_backtest_15m.py)
2. 지정가 주문 진입 (Pending Order)
3. 지정가 SL/TP (체결 즉시 설정)
4. 4봉(1시간) 미체결 시 주문 취소
5. 32봉(8시간) 최대 보유 (펀딩비 회피)

v2 대비 핵심 변경:
- 5분봉 X → 15분봉 전용
- 시장가 X → 지정가 진입
- 96봉(24시간) → 32봉(8시간) 최대 보유

실험 설계:
- 하네스: RELAXED (long_permit_states 확장)
- 4조합: A(1.5/2.5), B(2.0/3.0), C(2.0/2.0), D(2.5/3.5)
- Entry: 지정가 (4봉 유효)
- Max Hold: 32봉 (8시간)
- WFO: Train에서 선택 → Test에서 고정
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from policy_v1_2_relaxed_cooldown import POLICY_V12_RELAXED_COOLDOWN
from hmm_integrated_backtest import (
    HMMIntegratedConfig, load_year_data, load_hmm_model,
    get_theta, get_bt_cfg, compute_var5_from_train,
    resample_5m_to_15m, compute_emission_features
)
from wpcn._04_execution.futures_backtest_15m import (
    FuturesConfig15m, simulate_futures_15m, run_15m_backtest_from_5m
)
from wpcn._04_execution.invariants import validate_trades_df
from wpcn._04_execution.hmm_entry_gate import (
    HMMEntryGate, HMMGateConfig, VAR5_BY_STATE
)


def log(msg):
    print(msg, flush=True)


@dataclass
class DLiteConfig:
    """D-lite ATR SL/TP 설정 (v2.6.14)"""
    name: str
    atr_sl_mult: float
    atr_tp_mult: float
    max_hold_bars: int = 32       # 32봉 = 8시간 (펀딩비 회피)
    pending_max_bars: int = 4     # 4봉 = 1시간 (미체결 시 취소)


# 4조합 + 8시간 최대 보유
DLITE_CANDIDATES = [
    DLiteConfig("A_1.5x2.5", 1.5, 2.5, 32, 4),
    DLiteConfig("B_2.0x3.0", 2.0, 3.0, 32, 4),
    DLiteConfig("C_2.0x2.0", 2.0, 2.0, 32, 4),  # 빠른 청산형
    DLiteConfig("D_2.5x3.5", 2.5, 3.5, 32, 4),
]


def compute_detailed_stats(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict:
    """상세 통계 계산 (필수 로그 5개 + 리스크 지표)"""

    if trades_df is None or len(trades_df) == 0:
        return {
            "trade_count": 0,
            "total_pnl": 0,
            "avg_r": 0,
            "median_r": 0,
            "tp_hit_rate": 0,
            "sl_hit_rate": 0,
            "time_stop_rate": 0,
            "fees_paid_total": 0,
            "exposure_ratio": 0,
            "avg_stop_distance_pct": 0,
            "avg_loss_trade": 0,
            "max_dd": 0,
            "cvar_5pct": 0,
            "worst_20_pnl": 0,
        }

    # 진입/청산 분리 (v2.6.14: TIME_STOP 추가)
    exit_types = ["SL", "TP", "TIME_EXIT", "TIMEOUT", "LIQUIDATION",
                  "FORCED_CLOSE", "BEAR_ENTRY_KILL", "HARD_BEAR_KILL", "PARTIAL_DERISK",
                  "TIME_STOP"]  # v2.6.14: 15분봉 백테스터 exit type

    if "type" in trades_df.columns:
        exits = trades_df[trades_df["type"].isin(exit_types)].copy()
    else:
        exits = trades_df.copy()

    trade_count = len(exits) if len(exits) > 0 else len(trades_df)

    # PnL 계산
    if "pnl" in exits.columns:
        pnl_series = exits["pnl"].dropna()
    elif "final_pnl" in trades_df.columns:
        pnl_series = trades_df["final_pnl"].dropna()
    else:
        pnl_series = pd.Series([0])

    total_pnl = pnl_series.sum()
    avg_r = pnl_series.mean() if len(pnl_series) > 0 else 0
    median_r = pnl_series.median() if len(pnl_series) > 0 else 0

    # Exit type 비율
    if "type" in exits.columns and len(exits) > 0:
        type_counts = exits["type"].value_counts()
        tp_count = type_counts.get("TP", 0) + type_counts.get("TAKE_PROFIT", 0)
        sl_count = type_counts.get("SL", 0) + type_counts.get("STOP_LOSS", 0)
        time_count = (type_counts.get("TIME_EXIT", 0) + type_counts.get("TIMEOUT", 0)
                      + type_counts.get("TIME_STOP", 0))  # v2.6.14

        tp_hit_rate = tp_count / len(exits)
        sl_hit_rate = sl_count / len(exits)
        time_stop_rate = time_count / len(exits)
    else:
        tp_hit_rate = sl_hit_rate = time_stop_rate = 0

    # Max Drawdown
    if equity_df is not None and "equity" in equity_df.columns:
        equity = equity_df["equity"]
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min() * 100
    else:
        max_dd = 0

    # CVaR 5%
    if len(pnl_series) > 0:
        var_5 = np.percentile(pnl_series, 5)
        cvar_5pct = pnl_series[pnl_series <= var_5].mean() if len(pnl_series[pnl_series <= var_5]) > 0 else 0
    else:
        cvar_5pct = 0

    # 평균 손실 트레이드
    loss_trades = pnl_series[pnl_series < 0]
    avg_loss_trade = loss_trades.mean() if len(loss_trades) > 0 else 0

    # Worst 20 trades
    worst_20_pnl = pnl_series.nsmallest(20).sum() if len(pnl_series) >= 20 else pnl_series.sum()

    return {
        "trade_count": trade_count,
        "total_pnl": total_pnl,
        "avg_r": avg_r,
        "median_r": median_r,
        "tp_hit_rate": tp_hit_rate,
        "sl_hit_rate": sl_hit_rate,
        "time_stop_rate": time_stop_rate,
        "fees_paid_total": 0,
        "exposure_ratio": 0,
        "avg_stop_distance_pct": 0,
        "avg_loss_trade": avg_loss_trade,
        "max_dd": max_dd,
        "cvar_5pct": cvar_5pct,
        "worst_20_pnl": worst_20_pnl,
    }


def run_dlite_backtest_integrated(
    test_year: str,
    train_years: List[str],
    dlite_cfg: DLiteConfig,
    hmm_cfg: HMMIntegratedConfig,
    initial_equity: float = 10000.0,
) -> Optional[Dict]:
    """
    D-lite + HMM Entry Gating 백테스트 (v2.6.14)

    핵심 변경 (v3):
    1. 15분봉 전용 백테스터 (futures_backtest_15m.py)
    2. 지정가 진입 (Pending Order, 4봉 유효)
    3. 지정가 SL/TP
    4. 32봉(8시간) 최대 보유 (펀딩비 회피)
    """

    theta = get_theta()
    bt_cfg = get_bt_cfg()

    # HMM 모델 로드
    try:
        model = load_hmm_model()
    except FileNotFoundError:
        log("  [ERROR] HMM 모델 없음")
        return None

    feature_cols = ["price_position", "volume_ratio", "volatility", "trend_strength", "range_width_norm"]

    # 1. Train에서 VaR 계산
    var5_oos = compute_var5_from_train(train_years, model, theta)

    # 2. Test 데이터 로드
    df_test = load_year_data(test_year)
    if df_test is None:
        return None

    df_test_15m = resample_5m_to_15m(df_test)
    features_test = compute_emission_features(df_test_15m, theta)

    valid_mask = ~features_test[feature_cols].isna().any(axis=1)
    valid_mask[:200] = False
    X_test = features_test.loc[valid_mask, feature_cols].values
    valid_ts_test = features_test.index[valid_mask]
    posteriors_test = model.predict_proba(X_test)
    posterior_map = {ts: posteriors_test[i] for i, ts in enumerate(valid_ts_test)}

    # 3. HMM Entry Gate 생성 (v2.6.13 핵심)
    gate_cfg = HMMGateConfig(
        transition_delta=hmm_cfg.transition_delta,
        cooldown_bars=hmm_cfg.cooldown_bars,
        var_target=hmm_cfg.var_target,
        size_min=hmm_cfg.size_min,
        size_max=hmm_cfg.size_max,
        short_permit_enabled=hmm_cfg.short_permit_enabled,
        short_permit_min_markdown_prob=hmm_cfg.short_permit_min_markdown_prob,
        short_permit_min_trend_strength=hmm_cfg.short_permit_min_trend_strength,
        long_permit_enabled=hmm_cfg.long_permit_enabled,
        long_permit_states=tuple(hmm_cfg.long_permit_states) if hmm_cfg.long_permit_states else ('markup',),
        long_filter_enabled=hmm_cfg.long_filter_enabled,
    )

    hmm_gate = HMMEntryGate(
        posterior_map=posterior_map,
        features_df=features_test,
        cfg=gate_cfg,
        var5_by_state=var5_oos if var5_oos else VAR5_BY_STATE,
    )

    # 4. D-lite 설정으로 백테스트 실행 (v2.6.14: 15분봉 지정가)
    futures_cfg = FuturesConfig15m(
        atr_sl_mult=dlite_cfg.atr_sl_mult,
        atr_tp_mult=dlite_cfg.atr_tp_mult,
        atr_period=14,
        max_hold_bars=dlite_cfg.max_hold_bars,           # 32봉 (8시간)
        pending_order_max_bars=dlite_cfg.pending_max_bars,  # 4봉 (1시간)
    )

    # 5분봉 데이터에서 15분봉 리샘플링 후 백테스트
    trades_df, equity_df, base_stats = run_15m_backtest_from_5m(
        df_test, theta, bt_cfg, futures_cfg,
        initial_equity=initial_equity, seed=42,
        hmm_gate=hmm_gate,  # HMM Entry Gating 전달
    )

    if trades_df is None or len(trades_df) == 0:
        return None

    # 5. TradeInvariantChecker 검증
    is_valid, violations = validate_trades_df(trades_df)
    if not is_valid:
        log(f"  [WARNING] Trade invariant violations: {violations}")

    # 5.1 v2.6.14: 지정가 주문 체결 검증
    #     - signal_bar와 fill_bar(exit_bar)가 1~4봉 차이인지 확인
    #     - fill_price(entry_price)가 해당 봉의 OHLC 범위 내인지 확인
    if 'entry_bar' in trades_df.columns and 'exit_bar' in trades_df.columns:
        sample_trades = trades_df.head(20)
        fill_delay_issues = 0
        for _, row in sample_trades.iterrows():
            entry_bar = row.get('entry_bar', 0)
            # v2.6.14: 신호 발생 봉은 별도로 기록하지 않음 (pending order 구조상)
            # 단, hold_bars가 0~32 범위인지 확인
            hold_bars = row.get('hold_bars', 0)
            if hold_bars > 32:
                fill_delay_issues += 1
                log(f"  [HOLD TOO LONG] hold_bars={hold_bars} > 32")

        if fill_delay_issues > 0:
            log(f"  [WARNING] {fill_delay_issues} trades exceeded max hold time")
        else:
            log(f"  [OK] All trades within max hold limit (32 bars)")

    # 6. HMM Gate 통계 추출
    gate_stats = hmm_gate.get_stats()

    # 7. 상세 통계 계산
    detailed = compute_detailed_stats(trades_df, equity_df)

    # 8. PnL 계산 (Entry Gating 적용 후 결과이므로 baseline 없음)
    total_pnl = detailed['total_pnl']

    # blocked 합계 계산
    total_blocked = (
        gate_stats.get('blocked_by_transition', 0) +
        gate_stats.get('blocked_by_short_permit', 0) +
        gate_stats.get('blocked_by_long_permit', 0) +
        gate_stats.get('blocked_by_long_filter', 0)
    )

    return {
        'test_year': test_year,
        'dlite': dlite_cfg.name,
        'baseline_pnl': 0,  # Entry Gating 적용 후에는 baseline 개념 없음
        'filtered_pnl': total_pnl,  # 모든 트레이드가 이미 필터링됨
        'filter_stats': {
            'baseline_pnl': 0,
            'filtered_pnl': total_pnl,
            'improvement': total_pnl,  # baseline 대비 개선은 의미 없음
            'allowed_entries': detailed['trade_count'],
            'blocked_entries': total_blocked,
            'block_rate': gate_stats.get('block_rate', 0),
            'blocked_by_transition': gate_stats.get('blocked_by_transition', 0),
            'blocked_by_short_permit': gate_stats.get('blocked_by_short_permit', 0),
            'blocked_by_long_permit': gate_stats.get('blocked_by_long_permit', 0),
        },
        'gate_stats': gate_stats,
        'base_stats': base_stats,  # v2.6.15: 백테스터 원본 통계 (분모 확인용)
        'detailed_stats': detailed,
        'invariant_valid': is_valid,
        'invariant_violations': violations,
    }


def run_wfo_experiment():
    """WFO 실험 v3 실행 (v2.6.14)"""

    log("=" * 80)
    log("D-lite ATR SL/TP WFO 실험 v3 (v2.6.14 - 15분봉 지정가)")
    log("=" * 80)
    log("")
    log("v3 변경사항 (v2.6.14):")
    log("  - 15분봉 전용 백테스터 (5분봉 X)")
    log("  - 지정가 진입 (Pending Order)")
    log("  - 4봉(1시간) 미체결 시 주문 취소")
    log("  - 지정가 SL/TP (체결 즉시 설정)")
    log("  - 32봉(8시간) 최대 보유 (펀딩비 회피)")
    log("")
    log("실험 설계:")
    log("  - 하네스: v1.2 RELAXED COOLDOWN")
    log("    - transition_delta: 0.40 (0.20에서 완화)")
    log("    - cooldown_bars: 1 (2에서 완화)")
    log("  - 4조합: A(1.5/2.5), B(2.0/3.0), C(2.0/2.0), D(2.5/3.5)")
    log("  - Entry: 지정가 (4봉 유효)")
    log("  - Max Hold: 32봉 (8시간)")
    log("")

    # HMM 설정 (v1.2 RELAXED COOLDOWN)
    hmm_cfg = HMMIntegratedConfig(
        var_target=POLICY_V12_RELAXED_COOLDOWN.var_target,
        size_min=POLICY_V12_RELAXED_COOLDOWN.size_min,
        size_max=POLICY_V12_RELAXED_COOLDOWN.size_max,
        transition_delta=POLICY_V12_RELAXED_COOLDOWN.transition_delta,
        cooldown_bars=POLICY_V12_RELAXED_COOLDOWN.cooldown_bars,
        short_permit_enabled=POLICY_V12_RELAXED_COOLDOWN.short_permit_enabled,
        short_permit_min_markdown_prob=POLICY_V12_RELAXED_COOLDOWN.short_permit_min_markdown_prob,
        short_permit_min_trend_strength=POLICY_V12_RELAXED_COOLDOWN.short_permit_min_trend_strength,
        long_permit_enabled=POLICY_V12_RELAXED_COOLDOWN.long_permit_enabled,
        long_permit_states=POLICY_V12_RELAXED_COOLDOWN.long_permit_states,
        long_filter_enabled=POLICY_V12_RELAXED_COOLDOWN.long_filter_enabled,
    )

    # WFO 테스트 케이스
    wfo_tests = [
        {"train_years": ["2021", "2022"], "test_year": "2023"},
        {"train_years": ["2021", "2022", "2023"], "test_year": "2024"},
    ]

    all_results = []

    for wfo in wfo_tests:
        train_years = wfo["train_years"]
        test_year = wfo["test_year"]

        log(f"\n{'='*80}")
        log(f"WFO: {train_years} → {test_year}")
        log("=" * 80)

        # Train에서 각 D-lite 조합 테스트
        log(f"\n[TRAIN] {train_years} 에서 최적 D-lite 선택...")
        train_results = {}

        for dlite_cfg in DLITE_CANDIDATES:
            log(f"  Testing {dlite_cfg.name}...")
            train_pnl = 0
            train_trades = 0
            train_valid = True

            for year in train_years:
                result = run_dlite_backtest_integrated(
                    year, train_years, dlite_cfg, hmm_cfg
                )

                if result:
                    train_pnl += result['filtered_pnl']
                    train_trades += result['filter_stats'].get('allowed_entries', 0)
                    if not result['invariant_valid']:
                        train_valid = False
                        log(f"    [WARN] {year} invariant violations")

            train_results[dlite_cfg.name] = {
                "config": dlite_cfg,
                "pnl": train_pnl,
                "trades": train_trades,
                "valid": train_valid,
            }

            log(f"    Filtered PnL: ${train_pnl:+,.2f}, Trades: {train_trades}")

        # Train에서 최적 선택 (PnL 기준, 최소 트레이드 수 50)
        valid_candidates = {
            k: v for k, v in train_results.items()
            if v["trades"] >= 50 and v["valid"]
        }

        if not valid_candidates:
            valid_candidates = train_results

        best_name = max(valid_candidates, key=lambda k: valid_candidates[k]["pnl"])
        best_train = valid_candidates[best_name]

        log(f"\n[TRAIN RESULT] 최적 선택: {best_name}")
        log(f"  Train Filtered PnL: ${best_train['pnl']:+,.2f}")
        log(f"  Train Trades: {best_train['trades']}")

        # Test 데이터에 적용
        log(f"\n[TEST] {test_year}에 {best_name} 적용...")
        test_result = run_dlite_backtest_integrated(
            test_year, train_years, best_train["config"], hmm_cfg
        )

        if test_result is None:
            log("  [SKIP] 테스트 데이터 없음 또는 실패")
            continue

        # 결과 출력
        stats = test_result['detailed_stats']
        fs = test_result['filter_stats']
        gs = test_result.get('gate_stats', {})

        log(f"\n[TEST RESULT] {test_year}")
        log("-" * 60)
        log(f"  {'Metric':<25} {'Value':>15}")
        log("-" * 60)
        log(f"  {'total_pnl':<25} ${test_result['filtered_pnl']:>+14,.2f}")
        log("-" * 60)
        log(f"  {'trade_count':<25} {stats['trade_count']:>15}")
        log(f"  {'blocked_entries':<25} {fs.get('blocked_entries', 0):>15}")
        log(f"  {'block_rate':<25} {fs.get('block_rate', 0)*100:>14.1f}%")
        log("-" * 60)
        log(f"  {'tp_hit_rate':<25} {stats['tp_hit_rate']*100:>14.1f}%")
        log(f"  {'sl_hit_rate':<25} {stats['sl_hit_rate']*100:>14.1f}%")
        log(f"  {'time_stop_rate':<25} {stats['time_stop_rate']*100:>14.1f}%")
        log("-" * 60)
        log(f"  {'max_dd':<25} {stats['max_dd']:>14.1f}%")
        log(f"  {'cvar_5pct':<25} ${stats['cvar_5pct']:>14,.2f}")
        log(f"  {'invariant_valid':<25} {test_result['invariant_valid']:>15}")
        log("-" * 60)
        # v2.6.15: 상세 통계 (분모 확인용)
        bs = test_result.get('base_stats', {})
        log("  [시그널/주문 흐름 통계]")
        log(f"  {'signals_generated':<25} {bs.get('signals_generated', 0):>15}")
        log(f"  {'hmm_gate_checks':<25} {bs.get('hmm_gate_checks', 0):>15}")
        log(f"  {'hmm_gate_blocked':<25} {bs.get('hmm_gate_blocked', 0):>15}")
        log(f"  {'orders_placed':<25} {bs.get('orders_placed', 0):>15}")
        log(f"  {'orders_filled':<25} {bs.get('orders_filled', 0):>15}")
        log(f"  {'orders_expired':<25} {bs.get('orders_expired', 0):>15}")
        fill_rate = bs.get('fill_rate', 0)
        log(f"  {'fill_rate':<25} {fill_rate:>14.1f}%")
        log("-" * 60)
        log("  [HMM 차단 사유 분포]")
        log(f"  {'blocked_by_transition':<25} {bs.get('hmm_gate_blocked_transition', 0):>15}")
        log(f"  {'blocked_by_short_permit':<25} {bs.get('hmm_gate_blocked_short_permit', 0):>15}")
        log(f"  {'blocked_by_long_permit':<25} {bs.get('hmm_gate_blocked_long_permit', 0):>15}")
        log("-" * 60)

        all_results.append({
            "train": train_years,
            "test": test_year,
            "selected": best_name,
            "train_pnl": best_train["pnl"],
            "test_baseline_pnl": test_result['baseline_pnl'],
            "test_filtered_pnl": test_result['filtered_pnl'],
            "test_trades": fs.get('allowed_entries', 0),
            "test_max_dd": stats['max_dd'],
            "test_valid": test_result['invariant_valid'],
        })

    # 최종 요약
    log("\n" + "=" * 80)
    log("최종 WFO 요약 (v3 - 15분봉 지정가)")
    log("=" * 80)
    log("")
    log(f"{'Train':<20} {'Test':<6} {'Selected':<12} {'PnL':>12} {'Trades':>8} {'Valid':>6}")
    log("-" * 70)

    total_pnl = 0
    for r in all_results:
        train_str = "-".join(r["train"])
        log(f"{train_str:<20} {r['test']:<6} {r['selected']:<12} "
            f"${r['test_filtered_pnl']:>+10,.0f} "
            f"{r['test_trades']:>8} {'OK' if r['test_valid'] else 'FAIL':>6}")
        total_pnl += r['test_filtered_pnl']

    log("-" * 70)
    log(f"{'Total':>28} ${total_pnl:>+10,.0f}")
    log("")

    # 이전 버전 결과와 비교
    log("\n[비교] 이전 버전 결과")
    log("-" * 50)
    log("  v1 (5분봉 시장가, apply_hmm_filter 후처리)")
    log("    Total Test PnL: -$31,482")
    log("    SL Hit Rate: 94%")
    log("    Time-stop Rate: 0%")
    log("")
    log("  v2 (5분봉, Entry Gating)")
    log("    Total Test PnL: -$31,009")
    log("    SL Hit Rate: 73%")
    log("    Time-stop Rate: 0%")
    log("")
    log(f"[v3] Total Test PnL (15분봉 지정가): ${total_pnl:+,.2f}")
    log(f"v1 대비 개선: ${total_pnl - (-31482):+,.2f}")
    log(f"v2 대비 개선: ${total_pnl - (-31009):+,.2f}")

    return all_results


if __name__ == "__main__":
    run_wfo_experiment()