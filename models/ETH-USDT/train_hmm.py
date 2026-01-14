"""
ETH-USDT HMM Model Training
===========================

ETH 전용 6-State HMM 학습
- BTC와 동일한 아키텍처, ETH 데이터로 학습
- Sticky HMM (자기전이 강화)

사용법:
    python models/ETH-USDT/train_hmm.py

출력:
    models/ETH-USDT/hmm_model.pkl
    models/ETH-USDT/metrics.json
"""

import sys
import os
from pathlib import Path
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.optimize import linear_sum_assignment

from wpcn._03_common._01_core.types import Theta
from wpcn._04_execution.futures_backtest_v3 import resample_5m_to_15m
from wpcn.wyckoff.phases import detect_wyckoff_phase
from wpcn.wyckoff.box import box_engine_freeze
from wpcn.features.indicators import atr


# ============================================================================
# CONFIG
# ============================================================================
SYMBOL = "ETH-USDT"
DATA_PATH = Path("C:/Users/hahonggu/Desktop/coin_master/projects/trading_bot/data/bronze/binance/futures/ETH-USDT/5m")
OUTPUT_DIR = Path(__file__).parent

HMM_STATES = [
    'accumulation',
    're_accumulation',
    'distribution',
    're_distribution',
    'markup',
    'markdown',
]
STATE_TO_IDX = {s: i for i, s in enumerate(HMM_STATES)}
IDX_TO_STATE = {i: s for i, s in enumerate(HMM_STATES)}

UNCERTAINTY_THRESHOLD = 0.45
STICKY_KAPPA = 10.0


def get_theta() -> Theta:
    """ETH용 Theta (기본값으로 시작, 최적화 후 조정)"""
    return Theta(
        pivot_lr=5, box_L=24, atr_len=14, x_atr=0.5,
        m_bw=0.3, m_freeze=6, N_reclaim=4, N_fill=4, F_min=0.5,
    )


def load_eth_data(years: List[str] = None) -> pd.DataFrame:
    """ETH 5분봉 데이터 로드"""
    if years is None:
        years = ['2021', '2022', '2023', '2024']

    all_dfs = []
    for year in years:
        year_path = DATA_PATH / year
        if not year_path.exists():
            print(f"  [WARN] No data for {year}")
            continue

        files = sorted(year_path.glob("*.parquet"))
        for f in files:
            try:
                df_part = pd.read_parquet(f)
                if "timestamp" in df_part.columns:
                    df_part = df_part.set_index("timestamp")
                elif "open_time" in df_part.columns:
                    df_part['timestamp'] = pd.to_datetime(df_part['open_time'], unit='ms')
                    df_part = df_part.set_index('timestamp')
                all_dfs.append(df_part)
            except Exception as e:
                print(f"  [WARN] Failed to read {f}: {e}")

    if not all_dfs:
        return None

    df = pd.concat(all_dfs, ignore_index=False).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.index = pd.to_datetime(df.index)
    return df


def compute_emission_features(df: pd.DataFrame, theta: Theta) -> pd.DataFrame:
    """5차원 Emission Features 계산"""
    box = box_engine_freeze(df, theta)
    _atr = atr(df, theta.atr_len)
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    ema200 = df['close'].ewm(span=200, adjust=False).mean()
    vol_ma = df['volume'].rolling(20, min_periods=5).mean()

    features = pd.DataFrame(index=df.index)

    # 1. Price Position
    box_width = box['box_high'] - box['box_low']
    features['price_position'] = (df['close'] - box['box_low']) / (box_width + 1e-10)
    features['price_position'] = features['price_position'].clip(0, 1)

    # 2. Volume Ratio
    vol_ratio = df['volume'] / (vol_ma + 1e-10)
    features['volume_ratio_raw'] = np.log(vol_ratio + 1e-10)
    median = features['volume_ratio_raw'].rolling(100, min_periods=20).median()
    mad = (features['volume_ratio_raw'] - median).abs().rolling(100, min_periods=20).median()
    features['volume_ratio'] = (features['volume_ratio_raw'] - median) / (mad * 1.4826 + 1e-10)
    features['volume_ratio'] = features['volume_ratio'].clip(-3, 3)
    features = features.drop(columns=['volume_ratio_raw'])

    # 3. Volatility
    features['volatility'] = (_atr / df['close']) * 100
    features['volatility'] = features['volatility'].clip(0, 10)

    # 4. Trend Strength
    features['trend_strength'] = ((ema50 - ema200) / df['close']) * 100
    features['trend_strength'] = features['trend_strength'].clip(-10, 10)

    # 5. Range Width Norm
    features['range_width_norm'] = box_width / (_atr + 1e-10)
    features['range_width_norm'] = features['range_width_norm'].clip(0, 10)

    return features


def prepare_hmm_data(
    df_15m: pd.DataFrame,
    phases_df: pd.DataFrame,
    features: pd.DataFrame,
    min_lookback: int = 200
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """HMM 학습용 데이터 준비"""
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


def compute_initial_params(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """라벨 기반 초기 파라미터 계산"""
    n_states = len(HMM_STATES)
    n_features = X.shape[1]

    means = np.zeros((n_states, n_features))
    covars = np.zeros((n_states, n_features))

    for state, idx in STATE_TO_IDX.items():
        mask = (y == state)
        Xi = X[mask]
        if len(Xi) < 10:
            print(f"  [WARN] Too few samples for state {state}: {len(Xi)}")
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


def train_hmm(X: np.ndarray, y: np.ndarray, lengths: List[int], n_iter: int = 100, sticky: bool = True, kappa: float = STICKY_KAPPA):
    """Sticky HMM 학습"""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("[ERROR] hmmlearn not installed. Run: pip install hmmlearn")
        return None

    n_states = len(HMM_STATES)
    means, covars, transmat, startprob = compute_initial_params(X, y)

    if sticky:
        print(f"  [Sticky HMM] Applying kappa={kappa}...")
        for i in range(n_states):
            transmat[i, i] += kappa
        transmat = transmat / transmat.sum(axis=1, keepdims=True)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
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

    print(f"  Training with {len(X)} samples, {len(lengths)} segments...")
    model.fit(X, lengths)

    return model


def hungarian_matching(
    hmm_states: np.ndarray,
    rule_labels: np.ndarray,
    n_hmm_states: int = 6,
) -> Tuple[Dict[int, str], float]:
    """Hungarian algorithm으로 최적 매칭"""
    label_to_idx = {lbl: i for i, lbl in enumerate(HMM_STATES)}
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


def main():
    print("=" * 80)
    print(f"ETH-USDT HMM MODEL TRAINING")
    print("=" * 80)

    theta = get_theta()

    # 1. 데이터 로드
    print("\n[1] Loading ETH data...")
    df_5m = load_eth_data(['2021', '2022', '2023', '2024'])
    if df_5m is None:
        print("[ERROR] No data loaded")
        return

    print(f"    Loaded {len(df_5m):,} bars (5m)")

    df_15m = resample_5m_to_15m(df_5m)
    print(f"    Resampled to {len(df_15m):,} bars (15m)")

    # 2. Phase 탐지
    print("\n[2] Detecting Wyckoff phases...")
    phases_df = detect_wyckoff_phase(df_15m, theta)

    # 3. Emission Features
    print("\n[3] Computing emission features...")
    features = compute_emission_features(df_15m, theta)
    print(f"    Features shape: {features.shape}")

    # 4. HMM 데이터 준비
    print("\n[4] Preparing HMM data...")
    X, y, lengths = prepare_hmm_data(df_15m, phases_df, features)
    print(f"    X shape: {X.shape}")
    print(f"    Segments: {len(lengths)}")

    from collections import Counter
    label_counts = Counter(y)
    print(f"    Label distribution:")
    for state in HMM_STATES:
        count = label_counts.get(state, 0)
        pct = count / len(y) * 100
        print(f"      {state:<20}: {count:>8,} ({pct:>5.1f}%)")

    # 5. HMM 학습
    print("\n[5] Training Sticky HMM...")
    model = train_hmm(X, y, lengths, n_iter=100, sticky=True)
    if model is None:
        return

    print(f"    Log-likelihood: {model.score(X, lengths):.2f}")

    # 6. Hungarian Matching
    print("\n[6] Hungarian matching...")
    decoded_states = model.predict(X)
    mapping, accuracy = hungarian_matching(decoded_states, y)

    print(f"    Mapping:")
    for hmm_idx in sorted(mapping.keys()):
        print(f"      State {hmm_idx} -> {mapping[hmm_idx]}")
    print(f"    Accuracy: {accuracy:.1%}")

    # 7. 저장
    print("\n[7] Saving model...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 모델 저장
    with open(OUTPUT_DIR / "hmm_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # 매핑 저장
    with open(OUTPUT_DIR / "state_mapping.json", "w") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)

    # 메트릭 저장
    metrics = {
        'symbol': SYMBOL,
        'n_samples': len(y),
        'n_segments': len(lengths),
        'accuracy': float(accuracy),
        'log_likelihood': float(model.score(X, lengths)),
        'training_years': ['2021', '2022', '2023', '2024'],
        'created_at': datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n    Saved to: {OUTPUT_DIR}")
    print(f"      - hmm_model.pkl")
    print(f"      - state_mapping.json")
    print(f"      - metrics.json")

    print("\n" + "=" * 80)
    print(f"[OK] ETH-USDT HMM Model Training Complete!")
    print(f"    Accuracy: {accuracy:.1%}")
    print("=" * 80)

    return model, mapping, metrics


if __name__ == "__main__":
    result = main()
