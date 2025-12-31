"""
WPCN 통합 데이터 로더
Unified Data Loader

모든 백테스트 스크립트에서 공통으로 사용하는 데이터 로딩 모듈.
파일마다 복붙하던 로딩 코드를 하나로 통합.

주요 기능:
- 데이터 로딩 및 검증
- 중복 제거 및 정렬
- 타임존 처리
- 결측치 처리
- MTF 리샘플링

사용법:
    from wpcn.dataio.loader import DataLoader

    loader = DataLoader()
    df = loader.load_futures("BTC-USDT", "5m")
    df_15m = loader.resample(df, "15min")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from wpcn.config import get_data_path, PATHS

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """데이터 검증 오류"""
    pass


class DataLoader:
    """
    통합 데이터 로더

    FIXED:
    - except:pass 제거, 명시적 로깅
    - 데이터 검증 추가
    - 일관된 전처리
    """

    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

    def __init__(
        self,
        data_root: Optional[Path] = None,
        validate: bool = True
    ):
        """
        Args:
            data_root: 데이터 루트 경로 (기본: config에서 가져옴)
            validate: 데이터 검증 수행 여부
        """
        self.data_root = data_root or PATHS.DATA_DIR
        self.validate = validate

    def load_futures(
        self,
        symbol: str,
        timeframe: str = "5m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        선물 데이터 로드

        Args:
            symbol: 심볼 (예: "BTC-USDT")
            timeframe: 타임프레임 (예: "5m", "15m", "1h")
            start_date: 시작일 (선택)
            end_date: 종료일 (선택)

        Returns:
            DataFrame with DatetimeIndex or None if failed
        """
        data_path = get_data_path("bronze", "binance", "futures", symbol, timeframe)
        return self._load_parquet_dir(data_path, start_date, end_date)

    def load_spot(
        self,
        symbol: str,
        timeframe: str = "5m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """현물 데이터 로드"""
        data_path = get_data_path("bronze", "binance", "spot", symbol, timeframe)
        return self._load_parquet_dir(data_path, start_date, end_date)

    def _load_parquet_dir(
        self,
        data_path: Path,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Parquet 디렉토리에서 데이터 로드

        디렉토리 구조: data_path/year/month.parquet
        """
        if not data_path.exists():
            logger.warning(f"Data path not found: {data_path}")
            return None

        dfs = []
        files_loaded = 0
        files_failed = 0

        for year_dir in sorted(data_path.iterdir()):
            if not year_dir.is_dir():
                continue

            for month_file in sorted(year_dir.glob("*.parquet")):
                try:
                    df = pd.read_parquet(month_file)
                    dfs.append(df)
                    files_loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load {month_file}: {e}")
                    files_failed += 1

        if not dfs:
            logger.warning(f"No data files found in {data_path}")
            return None

        logger.info(f"Loaded {files_loaded} files, {files_failed} failed from {data_path}")

        # 병합
        df = pd.concat(dfs, ignore_index=True)

        # 전처리
        df = self._preprocess(df)

        # 날짜 필터링
        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        # 검증
        if self.validate:
            self._validate(df)

        return df

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 전처리

        - timestamp를 DatetimeIndex로 설정
        - 정렬 및 중복 제거
        - 필수 컬럼 확인
        """
        # timestamp 컬럼 찾기
        ts_col = None
        for col in ['timestamp', 'time', 'datetime', 'date']:
            if col in df.columns:
                ts_col = col
                break

        if ts_col is None:
            # 인덱스가 이미 DatetimeIndex인 경우
            if isinstance(df.index, pd.DatetimeIndex):
                pass
            else:
                raise DataValidationError("No timestamp column found")
        else:
            # timestamp를 인덱스로 설정
            df[ts_col] = pd.to_datetime(df[ts_col])
            df = df.set_index(ts_col)

        # 정렬
        df = df.sort_index()

        # 중복 제거 (같은 timestamp)
        df = df[~df.index.duplicated(keep='first')]

        # 컬럼명 소문자로 통일
        df.columns = df.columns.str.lower()

        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """
        데이터 검증

        Raises:
            DataValidationError: 검증 실패 시
        """
        # 인덱스 타입 확인
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError(
                f"Index must be DatetimeIndex, got {type(df.index)}"
            )

        # 인덱스 정렬 확인
        if not df.index.is_monotonic_increasing:
            raise DataValidationError("Index is not monotonic increasing")

        # 필수 컬럼 확인
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")

        # OHLCV 값 검증
        if (df['high'] < df['low']).any():
            bad_count = (df['high'] < df['low']).sum()
            logger.warning(f"Found {bad_count} rows where high < low")

        if (df['close'] < 0).any() or (df['open'] < 0).any():
            raise DataValidationError("Found negative prices")

        if (df['volume'] < 0).any():
            raise DataValidationError("Found negative volume")

    def resample(
        self,
        df: pd.DataFrame,
        freq: str,
        shift: bool = True
    ) -> pd.DataFrame:
        """
        OHLCV 데이터 리샘플링

        Args:
            df: 원본 DataFrame
            freq: 타겟 주기 (예: "15min", "1h", "4h")
            shift: True면 완성된 봉만 사용 (1봉 shift)

        Returns:
            리샘플링된 DataFrame
        """
        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        if shift:
            # 완성된 봉만 사용 (룩어헤드 방지)
            # shift(1)하면 현재 행에 이전 봉의 값이 들어감
            resampled = resampled.shift(1).dropna()

        return resampled

    def align_mtf(
        self,
        df_base: pd.DataFrame,
        df_higher: pd.DataFrame,
        columns: Optional[List[str]] = None,
        suffix: str = ""
    ) -> pd.DataFrame:
        """
        멀티타임프레임 정렬 (merge_asof 사용)

        상위 타임프레임 데이터를 하위 타임프레임에 정렬.
        룩어헤드 방지: direction='backward'로 과거 데이터만 참조.

        Args:
            df_base: 기준 DataFrame (예: 5분봉)
            df_higher: 상위 TF DataFrame (예: 15분봉, 이미 shift된 것)
            columns: 병합할 컬럼 (기본: 전체)
            suffix: 컬럼 접미사 (예: "_15m")

        Returns:
            병합된 DataFrame

        Example:
            >>> df_5m = loader.load_futures("BTC-USDT", "5m")
            >>> df_15m = loader.resample(df_5m, "15min", shift=True)
            >>> df_15m['stoch_k'], df_15m['stoch_d'] = stoch_rsi(df_15m['close'])
            >>> df_merged = loader.align_mtf(df_5m, df_15m[['stoch_k', 'stoch_d']], suffix="_15m")
        """
        # 인덱스를 컬럼으로 변환
        df_base_reset = df_base.reset_index()
        df_higher_reset = df_higher.reset_index()

        # 인덱스 컬럼명 통일
        base_time_col = df_base_reset.columns[0]
        higher_time_col = df_higher_reset.columns[0]

        df_higher_reset = df_higher_reset.rename(columns={higher_time_col: base_time_col})

        # 병합할 컬럼 선택
        if columns is not None:
            merge_cols = [base_time_col] + columns
            df_higher_reset = df_higher_reset[merge_cols]

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


# 편의 함수
def load_futures_data(
    symbol: str,
    timeframe: str = "5m",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Optional[pd.DataFrame]:
    """
    선물 데이터 로드 (편의 함수)

    Example:
        >>> df = load_futures_data("BTC-USDT", "5m")
    """
    loader = DataLoader()
    return loader.load_futures(symbol, timeframe, start_date, end_date)


def load_spot_data(
    symbol: str,
    timeframe: str = "5m",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Optional[pd.DataFrame]:
    """현물 데이터 로드 (편의 함수)"""
    loader = DataLoader()
    return loader.load_spot(symbol, timeframe, start_date, end_date)
