"""
데이터베이스 모델 및 테이블 스키마
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
import json


@dataclass
class Trade:
    """거래 기록"""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    tp_price: float
    sl_price: float
    pnl: Optional[float]
    pnl_pct: Optional[float]
    fee: float
    entry_time: datetime
    exit_time: Optional[datetime]
    exit_reason: Optional[str]  # 'tp', 'sl', 'timeout', 'manual'
    signal_score: float
    market_regime: Optional[str]  # 'bullish', 'bearish', 'ranging'
    timeframe: str
    strategy_version: str
    id: Optional[int] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Signal:
    """신호 로그"""
    symbol: str
    timeframe: str
    timestamp: datetime
    action: str  # 'long', 'short', 'none'
    score: float
    long_score: float
    short_score: float
    reason: dict  # 각 타임프레임별 점수 상세
    tf_alignment: int
    executed: bool
    id: Optional[int] = None

    def to_dict(self):
        d = asdict(self)
        d['reason'] = json.dumps(d['reason'])
        return d


@dataclass
class OptimizationResult:
    """주간 최적화 결과"""
    week: str  # '2024-W52'
    symbol: str
    timeframe: str
    params: dict
    backtest_result: dict
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    created_at: datetime
    id: Optional[int] = None

    def to_dict(self):
        d = asdict(self)
        d['params'] = json.dumps(d['params'])
        d['backtest_result'] = json.dumps(d['backtest_result'])
        return d


# 테이블 생성 SQL (INFORMATION_SCHEMA 사용 - 권한 문제 회피)
CREATE_TABLES_SQL = """
-- 거래 기록 테이블
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'trades')
CREATE TABLE trades (
    id INT IDENTITY(1,1) PRIMARY KEY,
    symbol NVARCHAR(20) NOT NULL,
    side NVARCHAR(10) NOT NULL,
    entry_price FLOAT NOT NULL,
    exit_price FLOAT NULL,
    quantity FLOAT NOT NULL,
    tp_price FLOAT NOT NULL,
    sl_price FLOAT NOT NULL,
    pnl FLOAT NULL,
    pnl_pct FLOAT NULL,
    fee FLOAT NOT NULL,
    entry_time DATETIME NOT NULL,
    exit_time DATETIME NULL,
    exit_reason NVARCHAR(20) NULL,
    signal_score FLOAT NOT NULL,
    market_regime NVARCHAR(20) NULL,
    timeframe NVARCHAR(10) NOT NULL,
    strategy_version NVARCHAR(20) NOT NULL,
    created_at DATETIME DEFAULT GETDATE()
);

-- 신호 로그 테이블
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'signals')
CREATE TABLE signals (
    id INT IDENTITY(1,1) PRIMARY KEY,
    symbol NVARCHAR(20) NOT NULL,
    timeframe NVARCHAR(10) NOT NULL,
    timestamp DATETIME NOT NULL,
    action NVARCHAR(10) NOT NULL,
    score FLOAT NOT NULL,
    long_score FLOAT NOT NULL,
    short_score FLOAT NOT NULL,
    reason NVARCHAR(MAX) NULL,
    tf_alignment INT NOT NULL,
    executed BIT NOT NULL,
    created_at DATETIME DEFAULT GETDATE()
);

-- 주간 최적화 결과 테이블
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'optimization_results')
CREATE TABLE optimization_results (
    id INT IDENTITY(1,1) PRIMARY KEY,
    week NVARCHAR(10) NOT NULL,
    symbol NVARCHAR(20) NOT NULL,
    timeframe NVARCHAR(10) NOT NULL,
    params NVARCHAR(MAX) NOT NULL,
    backtest_result NVARCHAR(MAX) NOT NULL,
    sharpe_ratio FLOAT NOT NULL,
    total_return FLOAT NOT NULL,
    max_drawdown FLOAT NOT NULL,
    win_rate FLOAT NOT NULL,
    created_at DATETIME NOT NULL
);
"""


def create_tables():
    """테이블 생성"""
    try:
        from .connection import get_cursor
    except ImportError:
        # 직접 실행 시
        import importlib.util
        from pathlib import Path
        conn_path = Path(__file__).parent / "connection.py"
        spec = importlib.util.spec_from_file_location("connection", conn_path)
        connection = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(connection)
        get_cursor = connection.get_cursor

    with get_cursor() as cursor:
        # 여러 문장을 개별 실행
        statements = [s.strip() for s in CREATE_TABLES_SQL.split(';') if s.strip()]
        for stmt in statements:
            try:
                cursor.execute(stmt)
            except Exception as e:
                print(f"테이블 생성 중 오류 (무시 가능): {e}")

    print("테이블 생성 완료")
