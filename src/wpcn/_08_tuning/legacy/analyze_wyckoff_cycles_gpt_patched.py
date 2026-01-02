"""
Wyckoff Cycle Length Analyzer

BTC 차트 데이터에서 Wyckoff 패턴의 사이클 길이를 분석하고
통계를 JSON 파일로 저장합니다.

분석 내용:
- Phase A → B → C 전체 사이클 길이 (봉 개수)
- 패턴 타입별 평균 사이클 (Accumulation, Distribution, Re-Accumulation, Re-Distribution)
- 타임프레임별 사이클 통계 (5m, 15m, 1h, 4h)
- FFT Cycle Detector와의 비교
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

from wpcn._00_config.config import PATHS, get_data_path, setup_logging
from wpcn._03_common._01_core.types import Theta
from wpcn._03_common._03_wyckoff.phases import detect_wyckoff_phase
from wpcn._03_common._02_features.dynamic_params import FFTCycleDetector

logger = setup_logging(level="INFO")

# ------------------------------
# Helpers: theta / robust metrics
# ------------------------------
def load_theta_from_project() -> Theta:
    """프로젝트 기본 Theta를 우선 사용하고, 없으면 ENV 기반으로 fallback."""
    # 1) 프로젝트에 이미 정의된 DEFAULT_THETA가 있다면 그걸 우선 사용
    for mod, name in [
        ("wpcn._00_config.defaults", "DEFAULT_THETA"),
        ("wpcn._00_config.config", "DEFAULT_THETA"),
        ("wpcn._00_config.theta", "DEFAULT_THETA"),
        ("wpcn._00_config.params", "DEFAULT_THETA"),
    ]:
        try:
            m = __import__(mod, fromlist=[name])
            val = getattr(m, name, None)
            if isinstance(val, Theta):
                return val
        except Exception:
            pass

    # 2) fallback: ENV 기반 (기존 동작)
    return Theta(
        pivot_lr=int(os.getenv("PIVOT_LR", "3")),
        box_L=int(os.getenv("BOX_LOOKBACK", "50")),
        m_freeze=int(os.getenv("M_FREEZE", "16")),
        atr_len=int(os.getenv("ATR_LEN", "14")),
        x_atr=float(os.getenv("X_ATR", "2.0")),
        m_bw=float(os.getenv("M_BW", "0.02")),
        N_reclaim=int(os.getenv("N_RECLAIM", "8")),
        N_fill=int(os.getenv("N_FILL", "5")),
        F_min=float(os.getenv("F_MIN", "0.3")),
    )


def _safe_std(x: List[float]) -> float:
    x = [v for v in x if np.isfinite(v)]
    if len(x) < 2:
        return 0.0
    return float(np.std(x, ddof=1))


def recommend_params(stats: Dict, top_3_cycles: List[Tuple[int, float]], theta: Theta) -> Dict:
    """통계 결과를 바탕으로 튜닝 참고용 추천 파라미터 산출."""
    overall = (stats or {}).get("overall") or {}
    complete_n = int(overall.get("complete_cycles", 0) or 0)

    # 데이터가 빈약하면 괜히 '추천'이라는 이름으로 거짓말하지 않음
    if complete_n < 3:
        return {
            "recommended_box_L": int(theta.box_L),
            "recommended_max_hold_bars": None,
            "recommended_fft_cycle": None,
            "note": "insufficient complete cycles (<3). keep defaults or widen lookback/days.",
        }

    median_len = float(overall.get("median_length", 0) or 0)
    p75_len = float(overall.get("percentile_75", 0) or 0)

    box_L = int(round(median_len)) if median_len > 0 else int(theta.box_L)
    max_hold = int(round(p75_len)) if p75_len > 0 else int(round(box_L * 1.1))

    fft_cycle = None
    if top_3_cycles:
        cycles_only = [int(c) for c, _ in top_3_cycles if c and np.isfinite(c)]
        if cycles_only:
            fft_cycle = min(cycles_only, key=lambda c: abs(c - box_L))

    return {
        "recommended_box_L": box_L,
        "recommended_max_hold_bars": max_hold,
        "recommended_fft_cycle": fft_cycle,
        "basis": {"median_cycle": median_len, "p75_cycle": p75_len, "complete_cycles": complete_n},
    }



def load_data(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """데이터 로드"""
    data_path = get_data_path("bronze", "binance", "futures", symbol, timeframe)

    if not data_path.exists():
        logger.warning(f"Data path not found: {data_path}")
        return None

    dfs = []
    for year_dir in data_path.iterdir():
        if not year_dir.is_dir():
            continue
        for month_file in year_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(month_file)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {month_file}: {e}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")

    # 최근 N일만 필터링
    cutoff_date = df.index.max() - pd.Timedelta(days=days)
    df = df[df.index >= cutoff_date]

    return df


def extract_wyckoff_cycles(phases_df: pd.DataFrame) -> List[Dict]:
    """
    Wyckoff Phase DataFrame에서 완전한 사이클 추출

    사이클 정의:
    - 연속된 박스 패턴을 하나의 사이클로 간주
    - Phase A/B/C 를 구분하지 않고 box가 형성되는 전체 구간을 측정
    - box_low와 box_high가 유효한 구간의 길이를 측정
    """
    cycles = []

    # box가 유효한지 확인 (box_low < box_high)
    valid_box = (phases_df['box_low'] < phases_df['box_high']) & \
                (phases_df['box_low'].notna()) & \
                (phases_df['box_high'].notna())

    # 박스 변화 감지 (스케일 불변: ATR / 박스폭 기반)
    # - 0.01 같은 절대값 임계는 BTC에선 의미 없고, 저가 알트에선 아예 안 잡힙니다.
    box_width = (phases_df['box_high'] - phases_df['box_low']).abs()

    # 환경변수로 조절 가능
    k_bw = float(os.getenv("BOX_SHIFT_THR_BW", "0.05"))    # 박스폭의 5%
    k_atr = float(os.getenv("BOX_SHIFT_THR_ATR", "0.50"))  # ATR의 0.5배 (atr 컬럼이 있을 때만)

    thr = box_width * k_bw

    if 'atr' in phases_df.columns:
        atr = phases_df['atr'].abs()
        thr = np.maximum(thr, atr * k_atr)

    # 너무 작은 thr로 박스가 잘게 쪼개지는 걸 방지 (수치 노이즈 방어)
    thr_floor = float(np.nanmedian(box_width) * 0.01) if np.isfinite(np.nanmedian(box_width)) else 0.0
    thr = np.maximum(thr, thr_floor)

    box_id = (phases_df['box_low'].diff().abs() > thr) | \
             (phases_df['box_high'].diff().abs() > thr)
    box_id = box_id.cumsum()

    # 각 박스 그룹별로 사이클 추출
    for box_num in box_id[valid_box].unique():
        mask = (box_id == box_num) & valid_box
        box_group = phases_df[mask]

        if len(box_group) < 10:  # 최소 10봉 이상만 유효한 사이클로 간주
            continue

        start_idx = box_group.index[0]
        end_idx = box_group.index[-1]
        length = len(box_group)

        # 주요 방향 결정 (가장 많이 나타난 direction)
        direction_counts = box_group['direction'].value_counts()
        main_direction = direction_counts.index[0] if len(direction_counts) > 0 else 'unknown'

        # 관찰된 phase들
        phases_seen = box_group['phase'].unique().tolist()
        sub_phases = box_group['sub_phase'].unique().tolist()

        # 평균 confidence
        avg_confidence = box_group['confidence'].mean()

        # Phase C가 있으면 완료, 없으면 미완료
        complete = 'C' in phases_seen

        cycle = {
            'start_idx': start_idx,
            'start_time': start_idx,
            'end_idx': end_idx,
            'end_time': end_idx,
            'length': length,
            'direction': main_direction,
            'complete': complete,
            'phases_seen': phases_seen,
            'sub_phases': sub_phases,
            'max_confidence': float(box_group['confidence'].max()),
            'avg_confidence': float(avg_confidence),
            'box_low': float(box_group['box_low'].iloc[0]),
            'box_high': float(box_group['box_high'].iloc[0])
        }

        cycles.append(cycle)

    return cycles


def calculate_cycle_statistics(cycles: List[Dict]) -> Dict:
    """사이클 통계 계산"""
    if not cycles:
        return {}

    # 완전한 사이클만 필터링
    complete_cycles = [c for c in cycles if c.get('complete', False)]

    # 전체 통계
    all_lengths = [c['length'] for c in complete_cycles]

    # 방향별 통계
    stats_by_direction = {}
    for direction in ['accumulation', 'distribution', 're_accumulation', 're_distribution']:
        dir_cycles = [c for c in complete_cycles if c['direction'] == direction]
        if dir_cycles:
            lengths = [c['length'] for c in dir_cycles]
            stats_by_direction[direction] = {
                'count': len(dir_cycles),
                'mean_length': float(np.mean(lengths)),
                'median_length': float(np.median(lengths)),
                'min_length': int(np.min(lengths)),
                'max_length': int(np.max(lengths)),
                'std_length': float(np.std(lengths)),
                'percentile_25': float(np.percentile(lengths, 25)),
                'percentile_75': float(np.percentile(lengths, 75))
            }

    # 전체 통계
    overall_stats = {
        'total_cycles': len(cycles),
        'complete_cycles': len(complete_cycles),
        'incomplete_cycles': len(cycles) - len(complete_cycles),
        'mean_length': float(np.mean(all_lengths)) if all_lengths else 0,
        'median_length': float(np.median(all_lengths)) if all_lengths else 0,
        'min_length': int(np.min(all_lengths)) if all_lengths else 0,
        'max_length': int(np.max(all_lengths)) if all_lengths else 0,
        'std_length': _safe_std(all_lengths) if all_lengths else 0,
        'percentile_25': float(np.percentile(all_lengths, 25)) if all_lengths else 0,
        'percentile_75': float(np.percentile(all_lengths, 75)) if all_lengths else 0
    }

    return {
        'overall': overall_stats,
        'by_direction': stats_by_direction
    }


def analyze_timeframe(symbol: str, timeframe: str, days: int, theta: Theta) -> Dict:
    """특정 타임프레임의 Wyckoff 사이클 분석"""
    print(f"\n{'='*80}")
    print(f"Analyzing {symbol} {timeframe} (last {days} days)")
    print(f"{'='*80}")

    # 데이터 로드
    df = load_data(symbol, timeframe, days)

    if df is None or len(df) < 200:
        print(f"  Insufficient data for {timeframe}")
        return None

    print(f"  Loaded {len(df):,} candles")
    print(f"  Period: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")

    # Wyckoff Phase 감지
    print(f"  Detecting Wyckoff phases...")
    phases_df = detect_wyckoff_phase(df, theta, lookback=50)

    # 사이클 추출
    print(f"  Extracting cycles...")
    cycles = extract_wyckoff_cycles(phases_df)

    # 통계 계산
    print(f"  Calculating statistics...")
    stats = calculate_cycle_statistics(cycles)

    # FFT Cycle Detection (비교용)
    print(f"  Running FFT cycle detection...")
    fft_detector = FFTCycleDetector(min_period=5, max_period=100)

    # FFT 입력: close 레벨 그대로 넣으면 추세가 스펙트럼을 오염시킵니다.
    # -> log return + Hann windowing + mean 제거(DC 제거)
    close = df['close'].to_numpy(dtype=float)
    lr = np.diff(np.log(close))
    lr = lr[np.isfinite(lr)]
    if len(lr) < 32:
        dominant_cycle, power = 0, 0.0
        top_3_cycles = []
    else:
        lr = lr - np.mean(lr)
        window = np.hanning(len(lr))
        x = lr * window

        dominant_cycle, power = fft_detector.detect_dominant_cycle(x)
        top_3_cycles = fft_detector.get_multiple_cycles(x, top_n=3)

    # 결과 출력
    if stats.get('overall'):
        overall = stats['overall']
        print(f"\n  Wyckoff Cycle Statistics:")
        print(f"    Total cycles detected: {overall['total_cycles']}")
        print(f"    Complete cycles: {overall['complete_cycles']}")
        print(f"    Average cycle length: {overall['mean_length']:.1f} bars")
        print(f"    Median cycle length: {overall['median_length']:.1f} bars")
        print(f"    Range: {overall['min_length']} - {overall['max_length']} bars")
        print(f"    25th-75th percentile: {overall['percentile_25']:.1f} - {overall['percentile_75']:.1f} bars")

        print(f"\n  By Direction:")
        for direction, dir_stats in stats.get('by_direction', {}).items():
            print(f"    {direction}: {dir_stats['count']} cycles, avg {dir_stats['mean_length']:.1f} bars")

    print(f"\n  FFT Dominant Cycle: {dominant_cycle} bars (power: {power:.3f})")
    print(f"  FFT Top 3 Cycles: {[(c, f'{p:.3f}') for c, p in top_3_cycles]}")

    # 추천 파라미터 (튜닝 참고용)
    reco = recommend_params(stats, [(int(c), float(p)) for c, p in top_3_cycles] if top_3_cycles else [], theta)

    return {
        'timeframe': timeframe,
        'symbol': symbol,
        'days_analyzed': days,
        'candles_count': len(df),
        'date_range': {
            'start': df.index.min().isoformat(),
            'end': df.index.max().isoformat()
        },
        'wyckoff_stats': stats,
        'recommended_params': reco,
        'fft_cycles': {
            'dominant_cycle': dominant_cycle,
            'dominant_power': float(power),
            'top_3_cycles': [(int(c), float(p)) for c, p in top_3_cycles]
        },
        'cycles_detail': [
            {
                'start_time': c.get('start_time', c['start_idx']).isoformat() if hasattr(c.get('start_time', c['start_idx']), 'isoformat') else str(c.get('start_time', c['start_idx'])),
                'end_time': c.get('end_time', c.get('end_idx', 'unknown')).isoformat() if hasattr(c.get('end_time', c.get('end_idx')), 'isoformat') else str(c.get('end_time', c.get('end_idx', 'unknown'))),
                'length': c.get('length', 0),
                'direction': c.get('direction', 'unknown'),
                'complete': c.get('complete', False),
                'phases_seen': c.get('phases_seen', []),
                'sub_phases': c.get('sub_phases', []),
                'max_confidence': c.get('max_confidence', 0.0)
            }
            for c in cycles
        ]
    }


def main():
    print("="*80)
    print("Wyckoff Cycle Length Analyzer")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 설정
    symbol = os.getenv("SYMBOL", "BTC-USDT")
    days = int(os.getenv("BACKTEST_DAYS", "90"))

    # Theta 설정 (프로젝트 기본값 우선)
    theta = load_theta_from_project()

    # 분석할 타임프레임들
    timeframes = ["15m", "1h", "4h"]

    results = {
        'analysis_date': datetime.now().isoformat(),
        'symbol': symbol,
        'days_analyzed': days,
        'theta_config': {
            'pivot_lr': theta.pivot_lr,
            'box_L': theta.box_L,
            'm_freeze': theta.m_freeze,
            'atr_len': theta.atr_len,
            'x_atr': theta.x_atr,
            'm_bw': theta.m_bw,
            'N_reclaim': theta.N_reclaim,
            'N_fill': theta.N_fill,
            'F_min': theta.F_min
        },
        'timeframes': {}
    }

    # 각 타임프레임 분석
    for tf in timeframes:
        result = analyze_timeframe(symbol, tf, days, theta)
        if result:
            results['timeframes'][tf] = result

    # JSON 저장
    output_file = PATHS.PROJECT_ROOT / "wyckoff_cycle_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

    # 요약 출력
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for tf, data in results['timeframes'].items():
        overall = data['wyckoff_stats'].get('overall', {})
        fft = data['fft_cycles']

        print(f"\n{tf.upper()} Timeframe:")
        print(f"  Wyckoff avg cycle: {overall.get('mean_length', 0):.1f} bars (median: {overall.get('median_length', 0):.1f})")
        print(f"  FFT dominant cycle: {fft['dominant_cycle']} bars")
        print(f"  Complete cycles found: {overall.get('complete_cycles', 0)}")

        by_dir = data['wyckoff_stats'].get('by_direction', {})
        if by_dir:
            print(f"  By direction:")
            for direction, stats in by_dir.items():
                print(f"    - {direction}: {stats['mean_length']:.1f} bars ({stats['count']} cycles)")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
