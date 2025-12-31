"""
WPCN 중앙 설정 모듈
Central Configuration Module

환경변수 또는 기본값으로 경로/설정을 관리합니다.
sys.path 하드코딩 제거를 위한 모듈.

사용법:
    from wpcn.config import PATHS, get_data_path

    data_dir = PATHS.DATA_DIR
    bronze_path = get_data_path("bronze", "binance", "futures")

환경변수:
    WPCN_DATA_DIR: 데이터 루트 디렉토리
    WPCN_RESULTS_DIR: 결과 저장 디렉토리
    WPCN_LOG_LEVEL: 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


def _get_project_root() -> Path:
    """프로젝트 루트 디렉토리 찾기"""
    # 이 파일 기준으로 상위로 올라가며 pyproject.toml 찾기
    current = Path(__file__).resolve().parent

    for _ in range(10):  # 최대 10단계 상위까지
        if (current / "pyproject.toml").exists():
            return current
        if current.parent == current:  # 루트에 도달
            break
        current = current.parent

    # 못 찾으면 현재 작업 디렉토리 사용
    return Path.cwd()


@dataclass
class PathConfig:
    """경로 설정"""
    PROJECT_ROOT: Path
    DATA_DIR: Path
    RESULTS_DIR: Path
    LOGS_DIR: Path

    @classmethod
    def from_env(cls) -> "PathConfig":
        """환경변수에서 경로 설정 로드"""
        project_root = _get_project_root()

        # 환경변수 우선, 없으면 기본값
        data_dir = Path(os.environ.get(
            "WPCN_DATA_DIR",
            str(project_root / "data")
        ))

        results_dir = Path(os.environ.get(
            "WPCN_RESULTS_DIR",
            str(project_root / "results")
        ))

        logs_dir = Path(os.environ.get(
            "WPCN_LOGS_DIR",
            str(project_root / "logs")
        ))

        return cls(
            PROJECT_ROOT=project_root,
            DATA_DIR=data_dir,
            RESULTS_DIR=results_dir,
            LOGS_DIR=logs_dir
        )


# 싱글톤 인스턴스
PATHS = PathConfig.from_env()


def get_data_path(*parts: str) -> Path:
    """
    데이터 경로 생성

    Args:
        *parts: 경로 구성요소 (예: "bronze", "binance", "futures")

    Returns:
        완성된 Path 객체

    Example:
        >>> get_data_path("bronze", "binance", "futures", "BTC-USDT", "5m")
        Path("data/bronze/binance/futures/BTC-USDT/5m")
    """
    return PATHS.DATA_DIR.joinpath(*parts)


def get_results_path(*parts: str) -> Path:
    """결과 경로 생성"""
    path = PATHS.RESULTS_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs() -> None:
    """필요한 디렉토리 생성"""
    PATHS.DATA_DIR.mkdir(parents=True, exist_ok=True)
    PATHS.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PATHS.LOGS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    로깅 설정

    Args:
        level: 로그 레벨 (기본: 환경변수 WPCN_LOG_LEVEL 또는 INFO)
        log_file: 로그 파일명 (기본: None = 콘솔만)

    Returns:
        설정된 루트 로거
    """
    if level is None:
        level = os.environ.get("WPCN_LOG_LEVEL", "INFO")

    log_level = getattr(logging, level.upper(), logging.INFO)

    # 포맷터
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 루트 로거 설정
    root_logger = logging.getLogger("wpcn")
    root_logger.setLevel(log_level)

    # 기존 핸들러 제거
    root_logger.handlers.clear()

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 파일 핸들러 (선택)
    if log_file:
        ensure_dirs()
        file_path = PATHS.LOGS_DIR / log_file
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


# 기본 심볼 목록
DEFAULT_FUTURES_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "XRP-USDT", "SOL-USDT", "DOGE-USDT",
    "LINK-USDT", "ZEC-USDT", "ICP-USDT", "FIL-USDT", "TRX-USDT", "BNB-USDT"
]

DEFAULT_SPOT_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "XRP-USDT", "SOL-USDT"
]
