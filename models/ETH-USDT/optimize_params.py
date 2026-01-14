"""
ETH-USDT Parameter Optimization
================================

ETH 전용 파라미터 최적화 (Grid Search)
- ATR SL/TP 배수
- 레버리지
- HMM Policy 파라미터

사용법:
    python models/ETH-USDT/optimize_params.py

출력:
    models/ETH-USDT/best_params.json
    trading_bot/config/symbols/ETH-USDT.yaml (업데이트)
"""

import sys
import os
from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from wpcn._03_common._01_core.types import Theta
from wpcn._04_execution.futures_backtest_15m import (
    FuturesConfig15m, simulate_futures_15m
)
from wpcn._04_execution.futures_backtest_v3 import resample_5m_to_15m
from wpcn._04_execution.hmm_entry_gate import HMMEntryGate, HMMGateConfig


# ============================================================================
# CONFIG
# ============================================================================
SYMBOL = "ETH-USDT"
DATA_PATH = Path("C:/Users/hahonggu/Desktop/coin_master/projects/trading_bot/data/bronze/binance/futures/ETH-USDT/5m")
MODEL_DIR = Path(__file__).parent
CONFIG_OUTPUT = Path("C:/Users/hahonggu/Desktop/coin_master/projects/trading_bot/config/symbols/ETH-USDT.yaml")


def log(msg):
    print(msg, flush=True)


def load_eth_5m(year: str) -> Optional[pd.DataFrame]:
    """ETH 5분봉 데이터 로드"""
    year_path = DATA_PATH / year
    if not year_path.exists():
        return None

    files = sorted(year_path.glob("*.parquet"))
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            elif "open_time" in df.columns:
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
                df = df.set_index('timestamp')
            dfs.append(df)
        except:
            pass

    if not dfs:
        return None

    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.index = pd.to_datetime(df.index)
    return df


def load_eth_hmm_model():
    """ETH HMM 모델 로드"""
    model_path = MODEL_DIR / "hmm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"ETH HMM model not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 매핑 로드
    mapping_path = MODEL_DIR / "state_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
            mapping = {int(k): v for k, v in mapping.items()}
    else:
        mapping = None

    return model, mapping


def get_theta() -> Theta:
    """ETH용 기본 Theta"""
    return Theta(
        pivot_lr=5, box_L=24, atr_len=14, x_atr=0.5,
        m_bw=0.3, m_freeze=6, N_reclaim=4, N_fill=4, F_min=0.5,
    )


def get_bt_cfg():
    """백테스트 기본 설정"""
    from wpcn._03_common._01_core.types import BacktestConfig
    return BacktestConfig()


def compute_emission_features(df: pd.DataFrame, theta: Theta) -> pd.DataFrame:
    """Emission Features 계산"""
    from wpcn.wyckoff.box import box_engine_freeze
    from wpcn.features.indicators import atr

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


def run_single_backtest(
    df_5m: pd.DataFrame,
    theta: Theta,
    bt_cfg,
    futures_cfg: FuturesConfig15m,
    hmm_gate: Optional[HMMEntryGate] = None,
    initial_equity: float = 10000.0,
) -> Dict:
    """단일 백테스트 실행"""
    from wpcn._04_execution.futures_backtest_15m import run_15m_backtest_from_5m

    trades_df, equity_df, stats = run_15m_backtest_from_5m(
        df_5m, theta, bt_cfg, futures_cfg,
        initial_equity=initial_equity, seed=42,
        hmm_gate=hmm_gate,
    )

    if trades_df is None:
        return None

    return stats


@dataclass
class ParamSet:
    """파라미터 조합"""
    atr_sl_mult: float
    atr_tp_mult: float
    leverage: float
    transition_delta: float = 0.40
    cooldown_bars: int = 1


def grid_search(
    df_5m: pd.DataFrame,
    model,
    theta: Theta,
    bt_cfg,
    feature_cols: List[str],
) -> Tuple[ParamSet, Dict]:
    """Grid Search로 최적 파라미터 찾기"""

    # 파라미터 그리드
    atr_sl_mults = [1.5, 2.0, 2.5, 3.0]
    atr_tp_mults = [2.0, 2.5, 3.0, 4.0]
    leverages = [3, 5, 10]

    log(f"\n[Grid Search] {len(atr_sl_mults) * len(atr_tp_mults) * len(leverages)} combinations")

    # 15분봉 변환 및 features 계산
    df_15m = resample_5m_to_15m(df_5m)
    features = compute_emission_features(df_15m, theta)

    valid_mask = ~features[feature_cols].isna().any(axis=1)
    valid_mask[:200] = False
    X = features.loc[valid_mask, feature_cols].values
    valid_ts = features.index[valid_mask]

    if len(X) == 0:
        log("  [ERROR] No valid features")
        return None, None

    posteriors = model.predict_proba(X)
    posterior_map = {ts: posteriors[i] for i, ts in enumerate(valid_ts)}

    # HMM Gate 설정
    from wpcn._04_execution.hmm_entry_gate import VAR5_BY_STATE

    best_pnl = float('-inf')
    best_params = None
    best_stats = None
    results = []

    for atr_sl in atr_sl_mults:
        for atr_tp in atr_tp_mults:
            if atr_tp <= atr_sl:  # TP는 SL보다 커야 함
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
                    posterior_map=posterior_map,
                    features_df=features,
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

                stats = run_single_backtest(
                    df_5m, theta, bt_cfg, futures_cfg,
                    hmm_gate=hmm_gate, initial_equity=10000.0
                )

                if stats is None:
                    continue

                pnl = stats.get('total_pnl', 0)
                trades = stats.get('total_trades', 0)
                win_rate = stats.get('win_rate', 0)

                results.append({
                    'atr_sl': atr_sl,
                    'atr_tp': atr_tp,
                    'leverage': lev,
                    'pnl': pnl,
                    'trades': trades,
                    'win_rate': win_rate,
                })

                if pnl > best_pnl and trades >= 30:  # 최소 30 trades
                    best_pnl = pnl
                    best_params = ParamSet(
                        atr_sl_mult=atr_sl,
                        atr_tp_mult=atr_tp,
                        leverage=lev,
                    )
                    best_stats = stats

    # 결과 출력
    log(f"\n[Grid Search Results] Top 10 by PnL:")
    results_sorted = sorted(results, key=lambda x: x['pnl'], reverse=True)[:10]
    for i, r in enumerate(results_sorted):
        log(f"  {i+1}. SL={r['atr_sl']:.1f} TP={r['atr_tp']:.1f} Lev={r['leverage']}x -> PnL=${r['pnl']:+,.2f} ({r['trades']} trades, {r['win_rate']:.1f}%)")

    return best_params, best_stats


def update_config_yaml(params: ParamSet, stats: Dict):
    """Config YAML 업데이트"""
    yaml_content = f"""# ETH-USDT Trading Parameters
# ============================
# 자동 최적화 결과 (v2.6.15)
# 최적화 날짜: {datetime.now().strftime('%Y-%m-%d %H:%M')}

symbol: ETH-USDT
enabled: true  # 최적화 완료 - 활성화

# Leverage - 최적화 결과
leverage:
  default: {int(params.leverage)}
  max: {int(min(params.leverage * 2, 25))}

# ETH 최적화된 실행 설정
execution:
  atr_sl_mult: {params.atr_sl_mult}
  atr_tp_mult: {params.atr_tp_mult}
  max_hold_bars: 32
  pending_order_max_bars: 4

# 백테스트 결과
backtest_stats:
  period: "2024"
  total_pnl: {stats.get('total_pnl', 0):.2f}
  trades: {stats.get('total_trades', 0)}
  win_rate: {stats.get('win_rate', 0):.1f}
  sl_rate: {stats.get('sl_rate', 0):.1f}
  tp_rate: {stats.get('tp_rate', 0):.1f}
  optimized_at: "{datetime.now().isoformat()}"
"""

    with open(CONFIG_OUTPUT, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    log(f"\n[Config Updated] {CONFIG_OUTPUT}")


def main():
    log("=" * 80)
    log("ETH-USDT PARAMETER OPTIMIZATION")
    log("=" * 80)

    # 1. HMM 모델 로드
    log("\n[1] Loading ETH HMM model...")
    try:
        model, mapping = load_eth_hmm_model()
        log(f"    Model loaded: {MODEL_DIR / 'hmm_model.pkl'}")
    except FileNotFoundError as e:
        log(f"    [ERROR] {e}")
        log("    Please run train_hmm.py first!")
        return

    theta = get_theta()
    bt_cfg = get_bt_cfg()
    feature_cols = ["price_position", "volume_ratio", "volatility", "trend_strength", "range_width_norm"]

    # 2. 테스트 데이터 로드
    log("\n[2] Loading test data (2024)...")
    df_5m = load_eth_5m("2024")
    if df_5m is None:
        log("    [ERROR] No 2024 data")
        return
    log(f"    Loaded {len(df_5m):,} bars")

    # 3. Grid Search
    log("\n[3] Running Grid Search...")
    best_params, best_stats = grid_search(
        df_5m, model, theta, bt_cfg, feature_cols
    )

    if best_params is None:
        log("    [ERROR] No valid parameter found")
        return

    log(f"\n[Best Parameters]")
    log(f"    ATR SL Mult: {best_params.atr_sl_mult}")
    log(f"    ATR TP Mult: {best_params.atr_tp_mult}")
    log(f"    Leverage: {best_params.leverage}x")
    log(f"    PnL: ${best_stats.get('total_pnl', 0):+,.2f}")

    # 4. 결과 저장
    log("\n[4] Saving results...")

    # best_params.json
    best_params_dict = {
        'symbol': SYMBOL,
        'atr_sl_mult': best_params.atr_sl_mult,
        'atr_tp_mult': best_params.atr_tp_mult,
        'leverage': best_params.leverage,
        'transition_delta': best_params.transition_delta,
        'cooldown_bars': best_params.cooldown_bars,
        'backtest_pnl': best_stats.get('total_pnl', 0),
        'backtest_trades': best_stats.get('total_trades', 0),
        'backtest_win_rate': best_stats.get('win_rate', 0),
        'optimized_at': datetime.now().isoformat(),
    }
    with open(MODEL_DIR / "best_params.json", "w") as f:
        json.dump(best_params_dict, f, indent=2)
    log(f"    Saved: {MODEL_DIR / 'best_params.json'}")

    # Config YAML 업데이트
    update_config_yaml(best_params, best_stats)

    log("\n" + "=" * 80)
    log("[OK] ETH-USDT Parameter Optimization Complete!")
    log("=" * 80)

    return best_params, best_stats


if __name__ == "__main__":
    result = main()
