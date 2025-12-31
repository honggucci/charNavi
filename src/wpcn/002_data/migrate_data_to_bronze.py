"""
데이터 마이그레이션 스크립트
raw/ 폴더의 parquet 파일들을 bronze/ 구조로 통합

구조: data/bronze/{거래소}/{심볼}/{타임프레임}/{년도}/{월}.parquet
"""

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 경로 설정
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
BRONZE_DIR = BASE_DIR / "data" / "bronze"


def parse_filename(filename: str) -> dict:
    """파일명에서 거래소, 심볼, 타임프레임 추출"""
    # binance_BTC-USDT_15m.parquet
    # binance_BTC-USDT_15m_365d.parquet
    # binance_BTC-USDT_5m_2024.parquet

    name = filename.replace(".parquet", "")
    parts = name.split("_")

    if len(parts) >= 3:
        exchange = parts[0]
        symbol = parts[1]
        timeframe = parts[2]
        return {
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe
        }
    return None


def load_and_merge_files():
    """모든 파일을 로드하고 심볼/타임프레임별로 그룹화"""

    # {(exchange, symbol, timeframe): [df1, df2, ...]}
    grouped_data = defaultdict(list)

    files = list(RAW_DIR.glob("*.parquet"))
    print(f"총 {len(files)}개 파일 발견\n")

    for file in files:
        info = parse_filename(file.name)
        if info is None:
            print(f"  [SKIP] 파싱 실패: {file.name}")
            continue

        key = (info["exchange"], info["symbol"], info["timeframe"])

        try:
            df = pd.read_parquet(file)
            grouped_data[key].append(df)
            print(f"  [OK] {file.name}: {len(df):,}행")
        except Exception as e:
            print(f"  [ERROR] {file.name}: {e}")

    return grouped_data


def merge_and_deduplicate(dfs: list) -> pd.DataFrame:
    """여러 DataFrame을 병합하고 중복 제거"""

    # index를 timestamp 컬럼으로 변환 (index가 datetime인 경우)
    processed_dfs = []
    for df in dfs:
        df = df.copy()
        # index가 datetime이면 컬럼으로 변환
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.columns = ["timestamp"] + list(df.columns[1:])
        elif df.index.name in ["time", "datetime", "date", "timestamp"]:
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "timestamp"})
        processed_dfs.append(df)

    # 모든 df 합치기
    merged = pd.concat(processed_dfs, ignore_index=True)

    # timestamp 컬럼 찾기
    timestamp_col = None
    for col in ["timestamp", "datetime", "date", "time", "open_time"]:
        if col in merged.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        # 첫 번째 컬럼이 timestamp일 가능성
        timestamp_col = merged.columns[0]

    # datetime 변환
    if not pd.api.types.is_datetime64_any_dtype(merged[timestamp_col]):
        merged[timestamp_col] = pd.to_datetime(merged[timestamp_col])

    # 중복 제거 (timestamp 기준)
    before_count = len(merged)
    merged = merged.drop_duplicates(subset=[timestamp_col], keep="last")
    after_count = len(merged)

    # 정렬
    merged = merged.sort_values(timestamp_col).reset_index(drop=True)

    print(f"    병합: {before_count:,}행 → 중복 제거 후: {after_count:,}행")

    return merged, timestamp_col


def save_by_month(df: pd.DataFrame, timestamp_col: str, exchange: str, symbol: str, timeframe: str):
    """월별로 분리하여 저장"""

    # 년/월 추출
    df["_year"] = df[timestamp_col].dt.year
    df["_month"] = df[timestamp_col].dt.month

    saved_files = []

    for (year, month), group in df.groupby(["_year", "_month"]):
        # 임시 컬럼 제거
        group = group.drop(columns=["_year", "_month"])

        # 경로 생성
        output_dir = BRONZE_DIR / exchange / symbol / timeframe / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{month:02d}.parquet"

        # 저장
        group.to_parquet(output_file, index=False)
        saved_files.append(f"{year}/{month:02d}.parquet ({len(group):,}행)")

    return saved_files


def main():
    print("=" * 60)
    print("데이터 마이그레이션: raw/ → bronze/")
    print("=" * 60)
    print()

    # 1. 파일 로드 및 그룹화
    print("[1/3] 파일 로드 중...")
    grouped_data = load_and_merge_files()
    print()

    # 2. 병합 및 저장
    print("[2/3] 병합 및 저장 중...")
    print()

    total_files = 0
    for (exchange, symbol, timeframe), dfs in grouped_data.items():
        print(f"  {exchange}/{symbol}/{timeframe}:")

        # 병합 및 중복 제거
        merged, timestamp_col = merge_and_deduplicate(dfs)

        # 월별 저장
        saved_files = save_by_month(merged, timestamp_col, exchange, symbol, timeframe)
        total_files += len(saved_files)

        for f in saved_files:
            print(f"    → {f}")
        print()

    # 3. 결과 출력
    print("[3/3] 완료!")
    print(f"  총 {total_files}개 파일 생성됨")
    print(f"  경로: {BRONZE_DIR}")
    print()
    print("=" * 60)
    print("기존 raw/ 폴더는 수동으로 확인 후 삭제하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
