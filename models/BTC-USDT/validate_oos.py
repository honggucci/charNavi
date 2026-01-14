"""
BTC-USDT Out-of-Sample Validation
==================================

Data Leakage 문제 해결:
- HMM 학습: 2021-2023 ONLY
- 파라미터 최적화: 2023 (validation)
- 최종 테스트: 2024 (NEVER SEEN)

사용법:
    python models/BTC-USDT/validate_oos.py
"""

import sys
import os
import glob
from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.optimize import linear_sum_assignment

from wpcn._03_common._01_core.types import Theta, BacktestConfig
from wpcn._04_execution.futures_backtest_15m import (
    FuturesConfig15m, simulate_futures_15m
)
from wpcn._04_execution.futures_backtest_v3 import resample_5m_to_15m
from wpcn._04_execution.hmm_entry_gate import HMMEntryGate, HMMGateConfig, VAR5_BY_STATE
from wpcn.wyckoff.phases import detect_wyckoff_phase
from wpcn.wyckoff.box import box_engine_freeze
from wpcn.features.indicators import atr


# ============================================================================
# CONFIG
# ============================================================================
SYMBOL = "BTC-USDT"
DATA_PATH = Path("C:/Users/hahonggu/Desktop/coin_master/projects/wpcn-backtester-cli-noflask/data/bronze/binance/futures/BTC-USDT/5m")
MODEL_DIR = Path(__file__).parent

HMM_STATES = ['accumulation', 're_accumulation', 'distribution', 're_distribution', 'markup', 'markdown']
STATE_TO_IDX = {s: i for i, s in enumerate(HMM_STATES)}
STICKY_KAPPA = 10.0


def log(msg):
    print(msg, flush=True)


def get_theta() -> Theta:
    return Theta(
        pivot_lr=5, box_L=24, atr_len=14, x_atr=0.5,
        m_bw=0.3, m_freeze=6, N_reclaim=4, N_fill=4, F_min=0.5,
    )


def load_btc_data(years: List[str]) -> Optional[pd.DataFrame]:
    """BTC 5분봉 데이터 로드"""
    all_dfs = []
    for year in years:
        pattern = f"{DATA_PATH}/{year}/*.parquet"
        files = sorted(glob.glob(pattern))
        for f in files:
            try:
                df_part = pd.read_parquet(f)
                if "timestamp" in df_part.columns:
                    df_part = df_part.set_index("timestamp")
                all_dfs.append(df_part)
            except Exception as e:
                pass

    if not all_dfs:
        return None

    df = pd.concat(all_dfs, ignore_index=False).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.index = pd.to_datetime(df.index)
    return df


def compute_emission_features(df: pd.DataFrame, theta: Theta) -> pd.DataFrame:
    """Emission Features 계산"""
    box = box_engine_freeze(df, theta)
    _atr = atr(df, theta.atr_len)
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    ema200 = df['close'].ewm(span=200, adjust=False).mean()
    vol_ma = df['volume'].rolling(20, min_periods=5).mean()

    features = pd.DataFrame(index=df.index)
    box_width = box['box_high'] - box['box_low']
    features['price_position'] = ((df['close'] - box['box_low']) / (box_width + 1e-10)).clip(0, 1)

    vol_ratio = df['volume'] / (vol_ma + 1e-10)
    vol_ratio_raw = np.log(vol_ratio + 1e-10)
    median = pd.Series(vol_ratio_raw).rolling(100, min_periods=20).median()
    mad = (pd.Series(vol_ratio_raw) - median).abs().rolling(100, min_periods=20).median()
    features['volume_ratio'] = ((vol_ratio_raw - median) / (mad * 1.4826 + 1e-10)).clip(-3, 3)
    features['volatility'] = ((_atr / df['close']) * 100).clip(0, 10)
    features['trend_strength'] = (((ema50 - ema200) / df['close']) * 100).clip(-10, 10)
    features['range_width_norm'] = (box_width / (_atr + 1e-10)).clip(0, 10)

    return features


def prepare_hmm_data(df_15m, phases_df, features, min_lookback=200):
    """HMM 학습 데이터 준비"""
    directions = phases_df['direction'].values
    feature_cols = ['price_position', 'volume_ratio', 'volatility', 'trend_strength', 'range_width_norm']

    valid_mask = np.isin(directions, HMM_STATES)
    valid_mask &= ~features[feature_cols].isna().any(axis=1).values
    valid_mask[:min_lookback] = False

    X = features.loc[valid_mask, feature_cols].values
    y = directions[valid_mask]

    lengths = []
    current_len = 0
    for v in valid_mask:
        if v:
            current_len += 1
        else:
            if current_len > 0:
                lengths.append(current_len)
            current_len = 0
    if current_len > 0:
        lengths.append(current_len)

    return X, y, lengths


def compute_initial_params(X, y, alpha=1.0):
    """초기 파라미터 계산"""
    n_states = len(HMM_STATES)
    n_features = X.shape[1]

    means = np.zeros((n_states, n_features))
    covars = np.zeros((n_states, n_features))

    for state, idx in STATE_TO_IDX.items():
        mask = (y == state)
        Xi = X[mask]
        if len(Xi) < 10:
            means[idx] = X.mean(axis=0)
            covars[idx] = X.var(axis=0) + 1e-4
        else:
            means[idx] = Xi.mean(axis=0)
            covars[idx] = Xi.var(axis=0) + 1e-4

    trans = np.zeros((n_states, n_states), dtype=float)
    for t in range(len(y) - 1):
        a, b = y[t], y[t+1]
        if a in STATE_TO_IDX and b in STATE_TO_IDX:
            trans[STATE_TO_IDX[a], STATE_TO_IDX[b]] += 1.0

    trans = trans + alpha
    trans = trans / trans.sum(axis=1, keepdims=True)
    startprob = np.ones(n_states) / n_states

    return means, covars, trans, startprob


def train_hmm_clean(X, y, lengths, kappa=STICKY_KAPPA):
    """Sticky HMM 학습 (Clean)"""
    from hmmlearn.hmm import GaussianHMM

    n_states = len(HMM_STATES)
    means, covars, transmat, startprob = compute_initial_params(X, y)

    for i in range(n_states):
        transmat[i, i] += kappa
    transmat = transmat / transmat.sum(axis=1, keepdims=True)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        tol=1e-4,
        random_state=42,
        init_params="",
        params="stmc",
        verbose=False,
    )

    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars
    model.fit(X, lengths)

    return model


def hungarian_matching(hmm_states, rule_labels):
    """Hungarian 매칭"""
    label_to_idx = {lbl: i for i, lbl in enumerate(HMM_STATES)}
    n_hmm_states = len(HMM_STATES)
    n_labels = len(HMM_STATES)

    confusion = np.zeros((n_hmm_states, n_labels), dtype=int)
    for hs, rl in zip(hmm_states, rule_labels):
        if rl in label_to_idx:
            confusion[hs, label_to_idx[rl]] += 1

    cost_matrix = -confusion
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = {}
    for hmm_idx, label_idx in zip(row_ind, col_ind):
        mapping[hmm_idx] = HMM_STATES[label_idx]

    remapped = np.array([mapping[s] for s in hmm_states])
    accuracy = (remapped == rule_labels).mean()

    return mapping, accuracy


def run_backtest(df_5m, theta, futures_cfg, hmm_gate=None, initial_equity=10000.0):
    """백테스트 실행"""
    from wpcn._04_execution.futures_backtest_15m import run_15m_backtest_from_5m
    bt_cfg = BacktestConfig()

    trades_df, equity_df, stats = run_15m_backtest_from_5m(
        df_5m, theta, bt_cfg, futures_cfg,
        initial_equity=initial_equity, seed=42,
        hmm_gate=hmm_gate,
    )

    return stats


def main():
    log("=" * 80)
    log("BTC-USDT OUT-OF-SAMPLE VALIDATION")
    log("=" * 80)
    log("\n[METHODOLOGY]")
    log("  - HMM Training:   2021-2023 ONLY (NO 2024)")
    log("  - Param Optimize: 2023 (validation set)")
    log("  - Final Test:     2024 (NEVER SEEN)")
    log("=" * 80)

    theta = get_theta()
    feature_cols = ['price_position', 'volume_ratio', 'volatility', 'trend_strength', 'range_width_norm']

    # ==========================================================================
    # STEP 1: Train HMM on 2021-2023 ONLY (NO 2024)
    # ==========================================================================
    log("\n[STEP 1] Training HMM on 2021-2023 ONLY...")

    df_train_5m = load_btc_data(['2021', '2022', '2023'])
    if df_train_5m is None:
        log("  [ERROR] No training data")
        return

    log(f"  Training data: {len(df_train_5m):,} bars (5m)")

    df_train_15m = resample_5m_to_15m(df_train_5m)
    log(f"  Resampled: {len(df_train_15m):,} bars (15m)")

    phases_train = detect_wyckoff_phase(df_train_15m, theta)
    features_train = compute_emission_features(df_train_15m, theta)
    X_train, y_train, lengths_train = prepare_hmm_data(df_train_15m, phases_train, features_train)

    log(f"  Training samples: {len(X_train):,}")

    model = train_hmm_clean(X_train, y_train, lengths_train)
    decoded_train = model.predict(X_train)
    mapping, train_accuracy = hungarian_matching(decoded_train, y_train)

    log(f"  HMM Accuracy (train): {train_accuracy:.1%}")

    # ==========================================================================
    # STEP 2: Optimize params on 2023 (validation)
    # ==========================================================================
    log("\n[STEP 2] Optimizing params on 2023 (validation)...")

    df_val_5m = load_btc_data(['2023'])
    df_val_15m = resample_5m_to_15m(df_val_5m)
    features_val = compute_emission_features(df_val_15m, theta)

    valid_mask = ~features_val[feature_cols].isna().any(axis=1)
    valid_mask[:200] = False
    X_val = features_val.loc[valid_mask, feature_cols].values
    valid_ts = features_val.index[valid_mask]

    posteriors_val = model.predict_proba(X_val)
    posterior_map_val = {ts: posteriors_val[i] for i, ts in enumerate(valid_ts)}

    # Grid Search on Validation Set
    atr_sl_mults = [1.5, 2.0, 2.5, 3.0]
    atr_tp_mults = [2.0, 2.5, 3.0, 4.0]
    leverages = [3, 5, 10]

    best_val_pnl = float('-inf')
    best_params = None
    val_results = []

    log(f"  Grid search: {len(atr_sl_mults) * len(atr_tp_mults) * len(leverages)} combinations")

    for atr_sl in atr_sl_mults:
        for atr_tp in atr_tp_mults:
            if atr_tp <= atr_sl:
                continue

            for lev in leverages:
                gate_cfg = HMMGateConfig(
                    transition_delta=0.40,
                    cooldown_bars=1,
                    var_target=-5.0,
                    size_min=0.5,
                    size_max=1.5,
                    short_permit_enabled=False,
                    long_permit_enabled=True,
                    long_permit_states=('markup', 'accumulation', 're_accumulation'),
                    long_filter_enabled=True,
                )

                hmm_gate = HMMEntryGate(
                    posterior_map=posterior_map_val,
                    features_df=features_val,
                    cfg=gate_cfg,
                    var5_by_state=VAR5_BY_STATE,
                )

                futures_cfg = FuturesConfig15m(
                    leverage=lev,
                    atr_sl_mult=atr_sl,
                    atr_tp_mult=atr_tp,
                    max_hold_bars=32,
                    pending_order_max_bars=4,
                )

                stats = run_backtest(df_val_5m, theta, futures_cfg, hmm_gate=hmm_gate)

                if stats is None:
                    continue

                pnl = stats.get('total_pnl', 0)
                trades = stats.get('total_trades', 0)
                win_rate = stats.get('win_rate', 0)

                val_results.append({
                    'atr_sl': atr_sl, 'atr_tp': atr_tp, 'leverage': lev,
                    'pnl': pnl, 'trades': trades, 'win_rate': win_rate
                })

                if pnl > best_val_pnl and trades >= 20:
                    best_val_pnl = pnl
                    best_params = {'atr_sl': atr_sl, 'atr_tp': atr_tp, 'leverage': lev}

    log(f"\n  [Validation Results] Top 5 by PnL:")
    val_sorted = sorted(val_results, key=lambda x: x['pnl'], reverse=True)[:5]
    for i, r in enumerate(val_sorted):
        log(f"    {i+1}. SL={r['atr_sl']:.1f} TP={r['atr_tp']:.1f} Lev={r['leverage']}x -> PnL=${r['pnl']:+,.2f} ({r['trades']} trades)")

    if best_params is None:
        log("  [ERROR] No valid params found")
        return

    log(f"\n  [Best Validation Params]: SL={best_params['atr_sl']} TP={best_params['atr_tp']} Lev={best_params['leverage']}x")
    log(f"  [Validation PnL]: ${best_val_pnl:+,.2f}")

    # ==========================================================================
    # STEP 3: Final Test on 2024 (NEVER SEEN)
    # ==========================================================================
    log("\n" + "=" * 80)
    log("[STEP 3] FINAL TEST ON 2024 (OUT-OF-SAMPLE)")
    log("=" * 80)

    df_test_5m = load_btc_data(['2024'])
    if df_test_5m is None:
        log("  [ERROR] No 2024 data")
        return

    log(f"  Test data: {len(df_test_5m):,} bars (5m)")

    df_test_15m = resample_5m_to_15m(df_test_5m)
    features_test = compute_emission_features(df_test_15m, theta)

    valid_mask_test = ~features_test[feature_cols].isna().any(axis=1)
    valid_mask_test[:200] = False
    X_test = features_test.loc[valid_mask_test, feature_cols].values
    valid_ts_test = features_test.index[valid_mask_test]

    posteriors_test = model.predict_proba(X_test)
    posterior_map_test = {ts: posteriors_test[i] for i, ts in enumerate(valid_ts_test)}

    # Create HMM Gate with test data
    gate_cfg = HMMGateConfig(
        transition_delta=0.40,
        cooldown_bars=1,
        var_target=-5.0,
        size_min=0.5,
        size_max=1.5,
        short_permit_enabled=False,
        long_permit_enabled=True,
        long_permit_states=('markup', 'accumulation', 're_accumulation'),
        long_filter_enabled=True,
    )

    hmm_gate_test = HMMEntryGate(
        posterior_map=posterior_map_test,
        features_df=features_test,
        cfg=gate_cfg,
        var5_by_state=VAR5_BY_STATE,
    )

    futures_cfg_test = FuturesConfig15m(
        leverage=best_params['leverage'],
        atr_sl_mult=best_params['atr_sl'],
        atr_tp_mult=best_params['atr_tp'],
        max_hold_bars=32,
        pending_order_max_bars=4,
    )

    log(f"\n  [Test Config]")
    log(f"    ATR SL Mult: {best_params['atr_sl']}")
    log(f"    ATR TP Mult: {best_params['atr_tp']}")
    log(f"    Leverage: {best_params['leverage']}x")

    test_stats = run_backtest(df_test_5m, theta, futures_cfg_test, hmm_gate=hmm_gate_test)

    if test_stats is None:
        log("  [ERROR] Backtest failed")
        return

    # ==========================================================================
    # RESULTS
    # ==========================================================================
    log("\n" + "=" * 80)
    log("FINAL RESULTS (OUT-OF-SAMPLE)")
    log("=" * 80)

    log(f"\n  Validation PnL (2023): ${best_val_pnl:+,.2f}")
    log(f"  Test PnL (2024):       ${test_stats.get('total_pnl', 0):+,.2f}")
    log(f"")
    log(f"  Test Statistics:")
    log(f"    Total Trades: {test_stats.get('total_trades', 0)}")
    log(f"    Win Rate:     {test_stats.get('win_rate', 0):.1f}%")
    log(f"    SL Rate:      {test_stats.get('sl_rate', 0):.1f}%")
    log(f"    TP Rate:      {test_stats.get('tp_rate', 0):.1f}%")
    log(f"    Max DD:       {test_stats.get('max_dd_pct', 0):.1f}%")

    # 비교: 기존 결과 vs OOS 결과
    # 기존 BTC 결과: 10x leverage에서 +$7,502
    original_leaked_pnl = 7502.0
    log(f"\n" + "-" * 40)
    log(f"COMPARISON:")
    log(f"-" * 40)
    log(f"  Original (Data Leakage): $+{original_leaked_pnl:,.0f}")
    log(f"  Out-of-Sample (2024):    ${test_stats.get('total_pnl', 0):+,.2f}")

    gap = original_leaked_pnl - test_stats.get('total_pnl', 0)
    log(f"  GAP (Overfitting):       ${gap:+,.2f}")

    # 결과 저장
    oos_results = {
        'symbol': SYMBOL,
        'methodology': 'Out-of-Sample',
        'train_years': ['2021', '2022', '2023'],
        'validation_year': '2023',
        'test_year': '2024',
        'best_params': best_params,
        'validation_pnl': best_val_pnl,
        'test_pnl': test_stats.get('total_pnl', 0),
        'test_trades': test_stats.get('total_trades', 0),
        'test_win_rate': test_stats.get('win_rate', 0),
        'test_sl_rate': test_stats.get('sl_rate', 0),
        'test_tp_rate': test_stats.get('tp_rate', 0),
        'original_leaked_pnl': original_leaked_pnl,
        'overfitting_gap': gap,
        'validated_at': datetime.now().isoformat(),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "oos_validation.json", "w") as f:
        json.dump(oos_results, f, indent=2)

    log(f"\n  [Saved] {MODEL_DIR / 'oos_validation.json'}")
    log("=" * 80)

    return oos_results


if __name__ == "__main__":
    result = main()