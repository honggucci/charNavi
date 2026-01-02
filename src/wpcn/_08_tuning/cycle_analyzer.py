"""
Wyckoff Cycle Length Analyzer v2
================================

BTC 차트 데이터에서 Wyckoff 패턴의 사이클 길이를 분석하고
통계를 JSON 파일로 저장합니다.

=== v2 수정사항 (Claude Code) ===
1. 박스 임계값: 0.01 고정 → ATR/box_width 기반 (스케일 불변)
2. FFT 입력: close 그대로 → log_return + Hann window (트렌드 오염 제거)
3. recommended_params: 튜닝 참고용 추천 파라미터 산출
4. Theta import: 실제 존재하는 모듈에서 직접 import
5. load_data: spot/futures 선택 가능
6. 사이클 완성 조건: C만 있으면 완성 → A→B→C 순서 검증
7. direction purity: 혼합도 지표 추가 (mixed 처리 가능)
8. std 계산: ddof=1 (표본 표준편차)

분석 내용:
- Phase A → B → C 전체 사이클 길이 (봉 개수)
- 패턴 타입별 평균 사이클 (Accumulation, Distribution, Re-Accumulation, Re-Distribution)
- 타임프레임별 사이클 통계 (15m, 1h, 4h)
- FFT Cycle Detector와의 비교

CLI 실행: python -m wpcn._08_tuning.cycle_analyzer
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
import os
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

from wpcn._00_config.config import PATHS, get_data_path, setup_logging
from wpcn._03_common._01_core.types import Theta
from wpcn._03_common._03_wyckoff.phases import detect_wyckoff_phase
from wpcn._03_common._02_features.dynamic_params import FFTCycleDetector

logger = setup_logging(level="INFO")


# ============================================================
# 프로젝트 기본 Theta (직접 정의)
# ============================================================
# GPT는 존재하지 않는 모듈 4개를 순회했음. 실제 프로젝트 구조 확인 결과:
# - DEFAULT_THETA는 types.py에 없고, Theta 클래스만 있음
# - 따라서 여기서 기본값 직접 정의 (백테스트 설정과 일치)
DEFAULT_THETA = Theta(
    pivot_lr=4,
    box_L=96,        # 15m*96 = 24시간
    m_freeze=32,
    atr_len=14,
    x_atr=0.35,      # GPT는 2.0으로 해놨음 (틀림)
    m_bw=0.10,       # GPT는 0.02로 해놨음 (틀림)
    N_reclaim=3,     # GPT는 8로 해놨음 (틀림)
    N_fill=5,
    F_min=0.70       # GPT는 0.3으로 해놨음 (틀림)
)


def load_theta() -> Theta:
    """
    Theta 로드 (프로젝트 기본값 우선, ENV fallback)

    GPT 버전과의 차이:
    - 존재하지 않는 모듈 순회 제거
    - 프로젝트 실제 기본값 사용 (types.py의 BacktestConfig와 일치)
    """
    # ENV 오버라이드가 있으면 사용
    if os.getenv("USE_ENV_THETA", "false").lower() == "true":
        return Theta(
            pivot_lr=int(os.getenv("PIVOT_LR", "4")),
            box_L=int(os.getenv("BOX_L", "96")),
            m_freeze=int(os.getenv("M_FREEZE", "32")),
            atr_len=int(os.getenv("ATR_LEN", "14")),
            x_atr=float(os.getenv("X_ATR", "0.35")),
            m_bw=float(os.getenv("M_BW", "0.10")),
            N_reclaim=int(os.getenv("N_RECLAIM", "3")),
            N_fill=int(os.getenv("N_FILL", "5")),
            F_min=float(os.getenv("F_MIN", "0.70")),
        )
    return DEFAULT_THETA


def load_data(symbol: str, timeframe: str, days: int, market: str = "spot") -> Optional[pd.DataFrame]:
    """
    데이터 로드

    GPT 버전과의 차이:
    - market 파라미터 추가 ("spot" / "futures")
    - 현물 분석인데 선물 데이터 읽는 문제 해결

    Args:
        symbol: 심볼 (예: "BTC-USDT")
        timeframe: 타임프레임 (예: "15m")
        days: 분석 기간 (일)
        market: "spot" 또는 "futures"
    """
    data_path = get_data_path("bronze", "binance", market, symbol, timeframe)

    if not data_path.exists():
        logger.warning(f"Data path not found: {data_path}")
        # fallback: 반대 마켓 시도
        alt_market = "futures" if market == "spot" else "spot"
        alt_path = get_data_path("bronze", "binance", alt_market, symbol, timeframe)
        if alt_path.exists():
            logger.info(f"Fallback to {alt_market} data: {alt_path}")
            data_path = alt_path
        else:
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


def validate_phase_sequence(phases_seen: List[str]) -> Tuple[bool, str]:
    """
    A → B → C 순서 검증 + sequence_type 반환

    GPT 버전: 'C' in phases_seen 만 확인 (순서 무시)
    v2: A가 B보다 먼저, B가 C보다 먼저 나와야 완성
    v2.1: sequence_type 추가 (A-B-C / A-C / B-C / C-only / incomplete)

    Returns:
        (is_complete, sequence_type)
        - is_complete: True if valid sequence exists
        - sequence_type: "A-B-C" | "A-C" | "B-C" | "C-only" | "incomplete"
    """
    if 'C' not in phases_seen:
        return False, "incomplete"

    # 순서 검증: A의 첫 등장 < B의 첫 등장 < C의 첫 등장
    try:
        first_a = phases_seen.index('A') if 'A' in phases_seen else -1
        first_b = phases_seen.index('B') if 'B' in phases_seen else -1
        first_c = phases_seen.index('C')

        # A와 B가 둘 다 있고, A < B < C 순서 (정석)
        if first_a >= 0 and first_b >= 0 and first_a < first_b < first_c:
            return True, "A-B-C"

        # A만 있고 B 없음: A < C면 허용 (Phase B가 짧아서 스킵된 경우)
        if first_a >= 0 and first_b < 0 and first_a < first_c:
            return True, "A-C"

        # A 없이 B, C만: B < C면 허용 (중간 진입)
        if first_a < 0 and first_b >= 0 and first_b < first_c:
            return True, "B-C"

        # C만 있음: 기술적으로 완성이지만 품질 낮음
        if first_a < 0 and first_b < 0:
            return False, "C-only"

        # 순서가 안 맞음
        return False, "incomplete"

    except (ValueError, IndexError):
        return False, "incomplete"


def calculate_direction_purity(direction_counts: pd.Series) -> Tuple[str, float]:
    """
    방향 purity 계산

    GPT 버전: 최빈값만 반환 (혼합도 무시)
    v2: purity 점수 추가, 낮으면 'mixed' 처리

    Returns:
        (direction, purity) - purity < 0.6이면 direction은 'mixed'
    """
    if len(direction_counts) == 0:
        return 'unknown', 0.0

    total = direction_counts.sum()
    top_count = direction_counts.iloc[0]
    purity = top_count / total if total > 0 else 0.0

    main_direction = direction_counts.index[0]

    # purity < 0.6이면 혼합 구간으로 판단
    if purity < 0.6:
        return 'mixed', purity

    return main_direction, purity


def extract_wyckoff_cycles(phases_df: pd.DataFrame, theta: Theta) -> List[Dict]:
    """
    Wyckoff Phase DataFrame에서 완전한 사이클 추출

    GPT 버전과의 차이:
    1. 박스 임계값: box_width * k 기반 (스케일 불변)
    2. 사이클 완성: A→B→C 순서 검증
    3. direction: purity 점수 포함
    """
    cycles = []

    # box가 유효한지 확인 (box_low < box_high)
    valid_box = (phases_df['box_low'] < phases_df['box_high']) & \
                (phases_df['box_low'].notna()) & \
                (phases_df['box_high'].notna())

    # ============================================================
    # 핵심 수정: 박스 변화 임계값 (스케일 불변)
    # ============================================================
    # GPT: box_width * 0.05 + atr * 0.5 + thr_floor
    # v2: 더 단순하게 box_width * k_bw, ATR은 있으면 max 처리

    box_width = (phases_df['box_high'] - phases_df['box_low']).abs()

    # 박스폭의 3% 이상 변화 시 새 박스 (더 보수적)
    k_bw = 0.03
    thr = box_width * k_bw

    # ATR이 있으면 추가 조건
    if 'atr' in phases_df.columns:
        atr = phases_df['atr'].abs()
        # ATR의 20% 이상 변화도 새 박스 조건에 추가
        thr = np.maximum(thr, atr * 0.2)

    # 최소 임계값 (수치 노이즈 방어)
    median_bw = np.nanmedian(box_width)
    if np.isfinite(median_bw) and median_bw > 0:
        thr_floor = median_bw * 0.005  # 0.5% 최소
        thr = np.maximum(thr, thr_floor)

    # 박스 변화 감지
    box_changed = (phases_df['box_low'].diff().abs() > thr) | \
                  (phases_df['box_high'].diff().abs() > thr)
    box_id = box_changed.cumsum()

    # 각 박스 그룹별로 사이클 추출
    for box_num in box_id[valid_box].unique():
        mask = (box_id == box_num) & valid_box
        box_group = phases_df[mask]

        if len(box_group) < 10:  # 최소 10봉 이상만 유효한 사이클로 간주
            continue

        start_idx = box_group.index[0]
        end_idx = box_group.index[-1]
        length = len(box_group)

        # 관찰된 phase들 (순서 유지)
        phases_seen = box_group['phase'].tolist()
        phases_unique = box_group['phase'].unique().tolist()
        sub_phases = box_group['sub_phase'].unique().tolist()

        # ============================================================
        # 핵심 수정: direction purity
        # ============================================================
        direction_counts = box_group['direction'].value_counts()
        main_direction, purity = calculate_direction_purity(direction_counts)

        # 평균 confidence
        avg_confidence = box_group['confidence'].mean()

        # ============================================================
        # 핵심 수정: A→B→C 순서 검증 + sequence_type
        # ============================================================
        complete, sequence_type = validate_phase_sequence(phases_seen)

        cycle = {
            'start_idx': start_idx,
            'start_time': start_idx,
            'end_idx': end_idx,
            'end_time': end_idx,
            'length': length,
            'direction': main_direction,
            'direction_purity': round(purity, 3),  # NEW: 혼합도
            'complete': complete,
            'sequence_type': sequence_type,  # NEW v2.1: A-B-C / A-C / B-C / C-only / incomplete
            'phases_seen': phases_unique,
            'phase_sequence_valid': complete,  # 순서 검증 결과
            'sub_phases': sub_phases,
            'max_confidence': float(box_group['confidence'].max()),
            'avg_confidence': float(avg_confidence),
            'box_low': float(box_group['box_low'].iloc[0]),
            'box_high': float(box_group['box_high'].iloc[0])
        }

        cycles.append(cycle)

    return cycles


def calculate_cycle_statistics(cycles: List[Dict]) -> Dict:
    """
    사이클 통계 계산

    GPT 버전과의 차이:
    - std 계산: ddof=1 (표본 표준편차)
    - purity 통계 추가
    """
    if not cycles:
        return {}

    # 완전한 사이클만 필터링
    complete_cycles = [c for c in cycles if c.get('complete', False)]

    # 높은 purity 사이클만 (purity >= 0.6)
    high_purity_cycles = [c for c in complete_cycles if c.get('direction_purity', 0) >= 0.6]

    # 정석 사이클만 (A-B-C sequence)
    abc_cycles = [c for c in complete_cycles if c.get('sequence_type') == 'A-B-C']

    # sequence_type별 카운트
    sequence_counts = {}
    for c in cycles:
        st = c.get('sequence_type', 'unknown')
        sequence_counts[st] = sequence_counts.get(st, 0) + 1

    # 전체 통계
    all_lengths = [c['length'] for c in complete_cycles]

    # 방향별 통계 (mixed 제외)
    stats_by_direction = {}
    for direction in ['accumulation', 'distribution', 're_accumulation', 're_distribution']:
        dir_cycles = [c for c in complete_cycles
                      if c['direction'] == direction and c.get('direction_purity', 0) >= 0.6]
        if dir_cycles:
            lengths = [c['length'] for c in dir_cycles]
            purities = [c.get('direction_purity', 0) for c in dir_cycles]
            stats_by_direction[direction] = {
                'count': len(dir_cycles),
                'mean_length': float(np.mean(lengths)),
                'median_length': float(np.median(lengths)),
                'min_length': int(np.min(lengths)),
                'max_length': int(np.max(lengths)),
                'std_length': float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0,
                'percentile_25': float(np.percentile(lengths, 25)),
                'percentile_75': float(np.percentile(lengths, 75)),
                'avg_purity': float(np.mean(purities))  # NEW
            }

    # 전체 통계
    overall_stats = {
        'total_cycles': len(cycles),
        'complete_cycles': len(complete_cycles),
        'abc_cycles': len(abc_cycles),  # NEW v2.1: 정석 A-B-C 사이클
        'high_purity_cycles': len(high_purity_cycles),
        'incomplete_cycles': len(cycles) - len(complete_cycles),
        'mixed_direction_cycles': len([c for c in cycles if c.get('direction') == 'mixed']),
        'sequence_type_counts': sequence_counts,  # NEW v2.1: 타입별 카운트
        'mean_length': float(np.mean(all_lengths)) if all_lengths else 0,
        'median_length': float(np.median(all_lengths)) if all_lengths else 0,
        'min_length': int(np.min(all_lengths)) if all_lengths else 0,
        'max_length': int(np.max(all_lengths)) if all_lengths else 0,
        'std_length': float(np.std(all_lengths, ddof=1)) if len(all_lengths) > 1 else 0,
        'percentile_25': float(np.percentile(all_lengths, 25)) if all_lengths else 0,
        'percentile_75': float(np.percentile(all_lengths, 75)) if all_lengths else 0
    }

    return {
        'overall': overall_stats,
        'by_direction': stats_by_direction
    }


def recommend_params(
    stats: Dict,
    top_3_cycles: List[Tuple[int, float]],
    theta: Theta,
    timeframe: str,
    peak_ratio: float = 0.0
) -> Dict:
    """
    통계 결과를 바탕으로 튜닝 참고용 추천 파라미터 산출

    GPT 버전과의 차이:
    - timeframe별 시간 환산 추가
    - 더 상세한 basis 정보
    """
    overall = (stats or {}).get("overall") or {}
    complete_n = int(overall.get("complete_cycles", 0) or 0)
    high_purity_n = int(overall.get("high_purity_cycles", 0) or 0)

    # 데이터가 빈약하면 추천 안함
    if complete_n < 3:
        return {
            "recommended_box_L": int(theta.box_L),
            "recommended_max_hold_bars": None,
            "recommended_fft_cycle": None,
            "confidence": "low",
            "note": f"insufficient complete cycles ({complete_n}<3). keep defaults.",
        }

    median_len = float(overall.get("median_length", 0) or 0)
    p75_len = float(overall.get("percentile_75", 0) or 0)

    box_L = int(round(median_len)) if median_len > 0 else int(theta.box_L)
    max_hold = int(round(p75_len)) if p75_len > 0 else int(round(box_L * 1.1))

    # FFT 주기 중 Wyckoff median과 가장 가까운 것 선택
    # v2.1: peak_ratio < 0.5면 FFT 추천 안함 (노이즈)
    fft_cycle = None
    fft_reliable = peak_ratio >= 0.5
    if top_3_cycles and fft_reliable:
        cycles_only = [int(c) for c, _ in top_3_cycles if c and np.isfinite(c)]
        if cycles_only:
            fft_cycle = min(cycles_only, key=lambda c: abs(c - box_L))

    # 시간 환산 (분 단위)
    tf_minutes = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    minutes = tf_minutes.get(timeframe, 15)

    box_L_hours = (box_L * minutes) / 60
    max_hold_hours = (max_hold * minutes) / 60

    # 신뢰도 판단
    confidence = "high" if high_purity_n >= 5 else "medium" if complete_n >= 5 else "low"

    return {
        "recommended_box_L": box_L,
        "recommended_max_hold_bars": max_hold,
        "recommended_fft_cycle": fft_cycle,
        "time_estimates": {
            "box_L_hours": round(box_L_hours, 1),
            "max_hold_hours": round(max_hold_hours, 1)
        },
        "confidence": confidence,
        "basis": {
            "median_cycle": median_len,
            "p75_cycle": p75_len,
            "complete_cycles": complete_n,
            "high_purity_cycles": high_purity_n
        },
    }


def run_fft_analysis(df: pd.DataFrame) -> Tuple[int, float, List[Tuple[int, float]], float]:
    """
    FFT 사이클 분석

    GPT 버전과의 차이:
    - 함수로 분리 (재사용성)
    - 더 안전한 예외 처리
    - v2.1: peak_ratio 추가 (FFT 신뢰도 게이트)

    Returns:
        (dominant_cycle, power, top_3_cycles, peak_ratio)
        - peak_ratio: top_power / sum_of_top3_power (0~1, 높을수록 dominant peak가 뚜렷)
    """
    fft_detector = FFTCycleDetector(min_period=5, max_period=100)

    close = df['close'].to_numpy(dtype=float)

    # log return 계산
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.diff(np.log(close))

    # 유효한 값만
    lr = lr[np.isfinite(lr)]

    if len(lr) < 64:  # 최소 64개 필요 (FFT 해상도)
        return 0, 0.0, [], 0.0

    # DC 제거 (mean 빼기)
    lr = lr - np.mean(lr)

    # Hann window 적용 (spectral leakage 방지)
    window = np.hanning(len(lr))
    x = lr * window

    try:
        dominant_cycle, power = fft_detector.detect_dominant_cycle(x)
        top_3_cycles = fft_detector.get_multiple_cycles(x, top_n=3)

        # ============================================================
        # NEW v2.1: peak_ratio 계산 (FFT 신뢰도)
        # ============================================================
        # peak_ratio = dominant_power / sum(top_3_powers)
        # 높을수록 dominant peak가 뚜렷함 (노이즈 적음)
        if top_3_cycles:
            sum_power = sum(p for _, p in top_3_cycles)
            peak_ratio = power / sum_power if sum_power > 0 else 0.0
        else:
            peak_ratio = 0.0

    except Exception as e:
        logger.warning(f"FFT analysis failed: {e}")
        return 0, 0.0, [], 0.0

    return dominant_cycle, power, top_3_cycles, peak_ratio


def analyze_timeframe(
    symbol: str,
    timeframe: str,
    days: int,
    theta: Theta,
    market: str = "spot"
) -> Optional[Dict]:
    """특정 타임프레임의 Wyckoff 사이클 분석"""
    print(f"\n{'='*80}")
    print(f"Analyzing {symbol} {timeframe} (last {days} days) [{market.upper()}]")
    print(f"{'='*80}")

    # 데이터 로드
    df = load_data(symbol, timeframe, days, market)

    if df is None or len(df) < 200:
        print(f"  Insufficient data for {timeframe}")
        return None

    print(f"  Loaded {len(df):,} candles")
    print(f"  Period: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")

    # Wyckoff Phase 감지
    print(f"  Detecting Wyckoff phases...")
    phases_df = detect_wyckoff_phase(df, theta, lookback=theta.box_L)

    # 사이클 추출
    print(f"  Extracting cycles...")
    cycles = extract_wyckoff_cycles(phases_df, theta)

    # 통계 계산
    print(f"  Calculating statistics...")
    stats = calculate_cycle_statistics(cycles)

    # FFT 분석
    print(f"  Running FFT cycle detection...")
    dominant_cycle, power, top_3_cycles, peak_ratio = run_fft_analysis(df)

    # 결과 출력
    if stats.get('overall'):
        overall = stats['overall']
        print(f"\n  Wyckoff Cycle Statistics:")
        print(f"    Total cycles detected: {overall['total_cycles']}")
        print(f"    Complete cycles: {overall['complete_cycles']}")
        print(f"    - A-B-C (정석): {overall.get('abc_cycles', 0)}")
        seq_counts = overall.get('sequence_type_counts', {})
        if seq_counts:
            print(f"    - A-C: {seq_counts.get('A-C', 0)}, B-C: {seq_counts.get('B-C', 0)}, C-only: {seq_counts.get('C-only', 0)}")
        print(f"    High purity cycles: {overall['high_purity_cycles']}")
        print(f"    Mixed direction cycles: {overall['mixed_direction_cycles']}")
        print(f"    Average cycle length: {overall['mean_length']:.1f} bars")
        print(f"    Median cycle length: {overall['median_length']:.1f} bars")
        print(f"    Range: {overall['min_length']} - {overall['max_length']} bars")
        print(f"    25th-75th percentile: {overall['percentile_25']:.1f} - {overall['percentile_75']:.1f} bars")

        print(f"\n  By Direction (purity >= 0.6):")
        for direction, dir_stats in stats.get('by_direction', {}).items():
            print(f"    {direction}: {dir_stats['count']} cycles, "
                  f"avg {dir_stats['mean_length']:.1f} bars, "
                  f"purity {dir_stats['avg_purity']:.2f}")

    # FFT 신뢰도 판단
    fft_reliable = peak_ratio >= 0.5  # 50% 이상이면 신뢰
    fft_status = "reliable" if fft_reliable else "noisy"

    print(f"\n  FFT Dominant Cycle: {dominant_cycle} bars (power: {power:.3f})")
    print(f"  FFT Peak Ratio: {peak_ratio:.2f} ({fft_status})")
    print(f"  FFT Top 3 Cycles: {[(c, f'{p:.3f}') for c, p in top_3_cycles]}")

    # 추천 파라미터
    top_3_typed = [(int(c), float(p)) for c, p in top_3_cycles] if top_3_cycles else []
    reco = recommend_params(stats, top_3_typed, theta, timeframe, peak_ratio)

    print(f"\n  Recommended Parameters:")
    print(f"    box_L: {reco['recommended_box_L']} bars ({reco.get('time_estimates', {}).get('box_L_hours', '?')}h)")
    print(f"    max_hold_bars: {reco['recommended_max_hold_bars']} bars ({reco.get('time_estimates', {}).get('max_hold_hours', '?')}h)")
    print(f"    Confidence: {reco['confidence']}")

    return {
        'timeframe': timeframe,
        'symbol': symbol,
        'market': market,
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
            'peak_ratio': round(peak_ratio, 3),  # NEW v2.1: FFT 신뢰도
            'reliable': fft_reliable,  # NEW v2.1: peak_ratio >= 0.5
            'top_3_cycles': top_3_typed
        },
        'cycles_detail': [
            {
                'start_time': c.get('start_time', c['start_idx']).isoformat()
                    if hasattr(c.get('start_time', c['start_idx']), 'isoformat')
                    else str(c.get('start_time', c['start_idx'])),
                'end_time': c.get('end_time', c.get('end_idx', 'unknown')).isoformat()
                    if hasattr(c.get('end_time', c.get('end_idx')), 'isoformat')
                    else str(c.get('end_time', c.get('end_idx', 'unknown'))),
                'length': c.get('length', 0),
                'direction': c.get('direction', 'unknown'),
                'direction_purity': c.get('direction_purity', 0),
                'complete': c.get('complete', False),
                'sequence_type': c.get('sequence_type', 'unknown'),  # NEW v2.1
                'phase_sequence_valid': c.get('phase_sequence_valid', False),
                'phases_seen': c.get('phases_seen', []),
                'sub_phases': c.get('sub_phases', []),
                'max_confidence': c.get('max_confidence', 0.0)
            }
            for c in cycles
        ]
    }


def run_analysis(
    symbol: str = "BTC-USDT",
    days: int = 90,
    market: str = "spot",
    timeframes: List[str] = None,
    theta: Theta = None,
    save_json: bool = True,
    output_path: str = None
) -> Dict[str, Any]:
    """
    프로그래밍 방식 분석 호출 (adaptive_space.py 연동용)

    Args:
        symbol: 심볼 (예: "BTC-USDT")
        days: 분석 기간 (일)
        market: "spot" 또는 "futures"
        timeframes: 분석할 타임프레임 리스트 (기본: ["15m", "1h", "4h"])
        theta: Theta 설정 (None이면 기본값)
        save_json: JSON 파일 저장 여부
        output_path: 저장 경로 (None이면 기본 경로)

    Returns:
        분석 결과 딕셔너리

    Example:
        >>> from analyze_wyckoff_cycles_v2 import run_analysis
        >>> result = run_analysis("BTC-USDT", days=90, market="spot")
        >>> from wpcn._08_tuning.adaptive_space import generate_adaptive_space
        >>> space = generate_adaptive_space(result, "15m")
    """
    if timeframes is None:
        timeframes = ["15m", "1h", "4h"]
    if theta is None:
        theta = load_theta()

    results = {
        'analysis_date': datetime.now().isoformat(),
        'version': 'v2_claude_code',
        'symbol': symbol,
        'market': market,
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

    for tf in timeframes:
        result = analyze_timeframe(symbol, tf, days, theta, market)
        if result:
            results['timeframes'][tf] = result

    if save_json:
        if output_path is None:
            output_path = PATHS.PROJECT_ROOT / "wyckoff_cycle_analysis_v2.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    print("="*80)
    print("Wyckoff Cycle Length Analyzer v2.1")
    print("(Claude Code Edition - GPT 패치 개선 + sequence_type + peak_ratio)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # 설정
    symbol = os.getenv("SYMBOL", "BTC-USDT")
    days = int(os.getenv("BACKTEST_DAYS", "90"))
    market = os.getenv("MARKET", "spot")  # NEW: spot/futures 선택

    # Theta 설정 (프로젝트 기본값)
    theta = load_theta()

    print(f"\n  Symbol: {symbol}")
    print(f"  Days: {days}")
    print(f"  Market: {market}")
    print(f"  Theta: box_L={theta.box_L}, x_atr={theta.x_atr}, m_bw={theta.m_bw}")

    # 분석할 타임프레임들
    timeframes = ["15m", "1h", "4h"]

    results = {
        'analysis_date': datetime.now().isoformat(),
        'version': 'v2_claude_code',
        'symbol': symbol,
        'market': market,
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
        'improvements_over_gpt': [
            "박스 임계값: 0.01→ATR/box_width 기반",
            "FFT: close→log_return+Hann window",
            "사이클 완성: C존재→A→B→C 순서검증",
            "direction: 최빈값→purity 점수 포함",
            "Theta: 존재하지않는모듈순회→직접정의",
            "load_data: futures고정→spot/futures선택",
            "std: population→sample(ddof=1)",
            "v2.1: sequence_type 저장 (A-B-C/A-C/B-C/C-only/incomplete)",
            "v2.1: FFT peak_ratio 저장 + 신뢰도 게이트 (noisy면 추천 안함)"
        ],
        'timeframes': {}
    }

    # 각 타임프레임 분석
    for tf in timeframes:
        result = analyze_timeframe(symbol, tf, days, theta, market)
        if result:
            results['timeframes'][tf] = result

    # JSON 저장
    output_file = PATHS.PROJECT_ROOT / "wyckoff_cycle_analysis_v2.json"
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
        reco = data['recommended_params']

        print(f"\n{tf.upper()} Timeframe:")
        print(f"  Wyckoff avg cycle: {overall.get('mean_length', 0):.1f} bars "
              f"(median: {overall.get('median_length', 0):.1f})")
        print(f"  FFT dominant cycle: {fft['dominant_cycle']} bars")
        print(f"  Complete cycles (A→B→C): {overall.get('complete_cycles', 0)}")
        print(f"  High purity: {overall.get('high_purity_cycles', 0)}")
        print(f"  ★ Recommended box_L: {reco.get('recommended_box_L', 'N/A')} bars "
              f"(confidence: {reco.get('confidence', 'N/A')})")

        by_dir = data['wyckoff_stats'].get('by_direction', {})
        if by_dir:
            print(f"  By direction:")
            for direction, dir_stats in by_dir.items():
                print(f"    - {direction}: {dir_stats['mean_length']:.1f} bars "
                      f"({dir_stats['count']} cycles, purity {dir_stats['avg_purity']:.2f})")

    print(f"\n{'='*80}")
    print("GPT야, 이게 진짜 튜닝 도구야. 네 버전은 리포트 생성기였어 ㅋ")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
