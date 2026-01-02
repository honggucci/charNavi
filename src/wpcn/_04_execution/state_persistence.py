"""
State Persistence Module (P0 - Crash Recovery)
==============================================

프로세스 재시작 시 포지션/Phase/주문 상태 복구를 위한 영속성 레이어.

Features:
- SQLite 기반 경량 저장소
- 포지션, 주문, 거래 기록 저장
- 마지막 처리된 캔들 타임스탬프 추적
- Idempotent 재시작 지원

Usage:
    from wpcn._04_execution.state_persistence import StateManager

    state = StateManager("BTC-USDT", "spot", "15m")

    # 저장
    state.save_position(side=1, qty=0.5, entry_price=50000, ...)
    state.save_last_candle_ts(pd.Timestamp("2025-01-01 12:00:00"))

    # 복구
    position = state.load_position()
    last_ts = state.load_last_candle_ts()
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import sqlite3
import json
import pandas as pd

from wpcn._00_config.config import PATHS


# ============================================================
# Data Classes
# ============================================================

@dataclass
class PersistedPosition:
    """영속화된 포지션 상태"""
    symbol: str
    market: str
    timeframe: str

    # 포지션 정보
    side: int  # +1 long, -1 short, 0 flat
    qty: float
    entry_price: float
    entry_time: Optional[str]

    # 청산 레벨
    stop_price: float
    tp1_price: float
    tp2_price: float
    tp1_exited: bool

    # MTF 포지션 (별도 추적)
    mtf_qty: float
    mtf_avg_entry: float

    # 축적 포지션
    accum_qty: float
    accum_avg_price: float

    # 메타
    active_box_high: float
    active_box_low: float
    entry_bar_index: int

    # 타임스탬프
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersistedPosition":
        return cls(**d)

    @classmethod
    def empty(cls, symbol: str, market: str, timeframe: str) -> "PersistedPosition":
        """빈 포지션 (flat)"""
        return cls(
            symbol=symbol,
            market=market,
            timeframe=timeframe,
            side=0,
            qty=0.0,
            entry_price=0.0,
            entry_time=None,
            stop_price=0.0,
            tp1_price=0.0,
            tp2_price=0.0,
            tp1_exited=False,
            mtf_qty=0.0,
            mtf_avg_entry=0.0,
            accum_qty=0.0,
            accum_avg_price=0.0,
            active_box_high=0.0,
            active_box_low=0.0,
            entry_bar_index=-1,
            updated_at=datetime.now().isoformat()
        )


@dataclass
class PersistedTrade:
    """거래 기록"""
    id: Optional[int]
    symbol: str
    market: str
    timeframe: str

    trade_type: str  # ENTRY, TP1, TP2, STOP, TIME_EXIT
    side: str  # long, short
    price: float
    qty: float
    pnl_pct: Optional[float]

    timestamp: str
    candle_time: str

    # 메타데이터 (JSON)
    meta: str


@dataclass
class PersistedState:
    """전체 상태 스냅샷"""
    symbol: str
    market: str
    timeframe: str

    # 마지막 처리된 캔들
    last_candle_ts: Optional[str]
    last_candle_index: int

    # 현재 활성 파라미터
    active_theta_run_id: Optional[str]
    active_theta_params: Optional[str]  # JSON

    # 현재 Phase
    current_phase: str  # A, B, C, D, E
    phase_direction: str  # accumulation, distribution

    # 캐시 (재계산 방지)
    cash: float

    updated_at: str


# ============================================================
# State Manager (SQLite)
# ============================================================

class StateManager:
    """
    상태 영속화 관리자

    SQLite 기반으로 포지션, 거래, 상태를 저장/복구
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        symbol: str,
        market: str = "spot",
        timeframe: str = "15m",
        db_path: Optional[Path] = None
    ):
        self.symbol = symbol
        self.market = market
        self.timeframe = timeframe

        # DB 경로 설정
        if db_path is None:
            db_dir = PATHS.PROJECT_ROOT / "data" / "state"
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = db_dir / f"state_{market}_{symbol.replace('-', '_')}.db"
        else:
            self.db_path = db_path

        # 테이블 초기화
        self._init_db()

    def _init_db(self):
        """데이터베이스 및 테이블 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 스키마 버전 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)

            # 포지션 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    market TEXT NOT NULL,
                    timeframe TEXT NOT NULL,

                    side INTEGER DEFAULT 0,
                    qty REAL DEFAULT 0.0,
                    entry_price REAL DEFAULT 0.0,
                    entry_time TEXT,

                    stop_price REAL DEFAULT 0.0,
                    tp1_price REAL DEFAULT 0.0,
                    tp2_price REAL DEFAULT 0.0,
                    tp1_exited INTEGER DEFAULT 0,

                    mtf_qty REAL DEFAULT 0.0,
                    mtf_avg_entry REAL DEFAULT 0.0,

                    accum_qty REAL DEFAULT 0.0,
                    accum_avg_price REAL DEFAULT 0.0,

                    active_box_high REAL DEFAULT 0.0,
                    active_box_low REAL DEFAULT 0.0,
                    entry_bar_index INTEGER DEFAULT -1,

                    updated_at TEXT NOT NULL,

                    UNIQUE(symbol, market, timeframe)
                )
            """)

            # 거래 기록 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    market TEXT NOT NULL,
                    timeframe TEXT NOT NULL,

                    trade_type TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    qty REAL NOT NULL,
                    pnl_pct REAL,

                    timestamp TEXT NOT NULL,
                    candle_time TEXT NOT NULL,
                    meta TEXT
                )
            """)

            # 상태 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    market TEXT NOT NULL,
                    timeframe TEXT NOT NULL,

                    last_candle_ts TEXT,
                    last_candle_index INTEGER DEFAULT -1,

                    active_theta_run_id TEXT,
                    active_theta_params TEXT,

                    current_phase TEXT DEFAULT 'unknown',
                    phase_direction TEXT DEFAULT 'unknown',

                    cash REAL DEFAULT 10000.0,

                    updated_at TEXT NOT NULL,

                    UNIQUE(symbol, market, timeframe)
                )
            """)

            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_time
                ON trades(symbol, market, timeframe, timestamp)
            """)

            conn.commit()

    # ============================================================
    # Position CRUD
    # ============================================================

    def save_position(self, position: PersistedPosition) -> None:
        """포지션 저장 (upsert)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO positions (
                    symbol, market, timeframe,
                    side, qty, entry_price, entry_time,
                    stop_price, tp1_price, tp2_price, tp1_exited,
                    mtf_qty, mtf_avg_entry,
                    accum_qty, accum_avg_price,
                    active_box_high, active_box_low, entry_bar_index,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, market, timeframe) DO UPDATE SET
                    side = excluded.side,
                    qty = excluded.qty,
                    entry_price = excluded.entry_price,
                    entry_time = excluded.entry_time,
                    stop_price = excluded.stop_price,
                    tp1_price = excluded.tp1_price,
                    tp2_price = excluded.tp2_price,
                    tp1_exited = excluded.tp1_exited,
                    mtf_qty = excluded.mtf_qty,
                    mtf_avg_entry = excluded.mtf_avg_entry,
                    accum_qty = excluded.accum_qty,
                    accum_avg_price = excluded.accum_avg_price,
                    active_box_high = excluded.active_box_high,
                    active_box_low = excluded.active_box_low,
                    entry_bar_index = excluded.entry_bar_index,
                    updated_at = excluded.updated_at
            """, (
                position.symbol, position.market, position.timeframe,
                position.side, position.qty, position.entry_price, position.entry_time,
                position.stop_price, position.tp1_price, position.tp2_price, int(position.tp1_exited),
                position.mtf_qty, position.mtf_avg_entry,
                position.accum_qty, position.accum_avg_price,
                position.active_box_high, position.active_box_low, position.entry_bar_index,
                position.updated_at
            ))

            conn.commit()

    def load_position(self) -> Optional[PersistedPosition]:
        """포지션 로드"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    symbol, market, timeframe,
                    side, qty, entry_price, entry_time,
                    stop_price, tp1_price, tp2_price, tp1_exited,
                    mtf_qty, mtf_avg_entry,
                    accum_qty, accum_avg_price,
                    active_box_high, active_box_low, entry_bar_index,
                    updated_at
                FROM positions
                WHERE symbol = ? AND market = ? AND timeframe = ?
            """, (self.symbol, self.market, self.timeframe))

            row = cursor.fetchone()
            if row is None:
                return None

            return PersistedPosition(
                symbol=row[0],
                market=row[1],
                timeframe=row[2],
                side=row[3],
                qty=row[4],
                entry_price=row[5],
                entry_time=row[6],
                stop_price=row[7],
                tp1_price=row[8],
                tp2_price=row[9],
                tp1_exited=bool(row[10]),
                mtf_qty=row[11],
                mtf_avg_entry=row[12],
                accum_qty=row[13],
                accum_avg_price=row[14],
                active_box_high=row[15],
                active_box_low=row[16],
                entry_bar_index=row[17],
                updated_at=row[18]
            )

    def clear_position(self) -> None:
        """포지션 초기화 (flat으로 설정)"""
        empty = PersistedPosition.empty(self.symbol, self.market, self.timeframe)
        self.save_position(empty)

    # ============================================================
    # Trade Log
    # ============================================================

    def log_trade(
        self,
        trade_type: str,
        side: str,
        price: float,
        qty: float,
        candle_time: pd.Timestamp,
        pnl_pct: Optional[float] = None,
        meta: Optional[Dict] = None
    ) -> int:
        """거래 기록 추가"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO trades (
                    symbol, market, timeframe,
                    trade_type, side, price, qty, pnl_pct,
                    timestamp, candle_time, meta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.symbol, self.market, self.timeframe,
                trade_type, side, price, qty, pnl_pct,
                datetime.now().isoformat(),
                str(candle_time),
                json.dumps(meta or {})
            ))

            conn.commit()
            return cursor.lastrowid

    def get_recent_trades(self, limit: int = 50) -> List[PersistedTrade]:
        """최근 거래 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, symbol, market, timeframe,
                       trade_type, side, price, qty, pnl_pct,
                       timestamp, candle_time, meta
                FROM trades
                WHERE symbol = ? AND market = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (self.symbol, self.market, self.timeframe, limit))

            trades = []
            for row in cursor.fetchall():
                trades.append(PersistedTrade(
                    id=row[0],
                    symbol=row[1],
                    market=row[2],
                    timeframe=row[3],
                    trade_type=row[4],
                    side=row[5],
                    price=row[6],
                    qty=row[7],
                    pnl_pct=row[8],
                    timestamp=row[9],
                    candle_time=row[10],
                    meta=row[11]
                ))

            return trades

    def check_duplicate_trade(
        self,
        trade_type: str,
        candle_time: pd.Timestamp
    ) -> bool:
        """중복 거래 체크 (idempotent 보장)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM trades
                WHERE symbol = ? AND market = ? AND timeframe = ?
                AND trade_type = ? AND candle_time = ?
            """, (
                self.symbol, self.market, self.timeframe,
                trade_type, str(candle_time)
            ))

            count = cursor.fetchone()[0]
            return count > 0

    # ============================================================
    # State (Last Candle, Theta, Phase)
    # ============================================================

    def save_state(
        self,
        last_candle_ts: Optional[pd.Timestamp] = None,
        last_candle_index: int = -1,
        active_theta_run_id: Optional[str] = None,
        active_theta_params: Optional[Dict] = None,
        current_phase: str = "unknown",
        phase_direction: str = "unknown",
        cash: float = 10000.0
    ) -> None:
        """전체 상태 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO state (
                    symbol, market, timeframe,
                    last_candle_ts, last_candle_index,
                    active_theta_run_id, active_theta_params,
                    current_phase, phase_direction,
                    cash, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, market, timeframe) DO UPDATE SET
                    last_candle_ts = excluded.last_candle_ts,
                    last_candle_index = excluded.last_candle_index,
                    active_theta_run_id = excluded.active_theta_run_id,
                    active_theta_params = excluded.active_theta_params,
                    current_phase = excluded.current_phase,
                    phase_direction = excluded.phase_direction,
                    cash = excluded.cash,
                    updated_at = excluded.updated_at
            """, (
                self.symbol, self.market, self.timeframe,
                str(last_candle_ts) if last_candle_ts else None,
                last_candle_index,
                active_theta_run_id,
                json.dumps(active_theta_params) if active_theta_params else None,
                current_phase, phase_direction,
                cash,
                datetime.now().isoformat()
            ))

            conn.commit()

    def load_state(self) -> Optional[PersistedState]:
        """전체 상태 로드"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    symbol, market, timeframe,
                    last_candle_ts, last_candle_index,
                    active_theta_run_id, active_theta_params,
                    current_phase, phase_direction,
                    cash, updated_at
                FROM state
                WHERE symbol = ? AND market = ? AND timeframe = ?
            """, (self.symbol, self.market, self.timeframe))

            row = cursor.fetchone()
            if row is None:
                return None

            return PersistedState(
                symbol=row[0],
                market=row[1],
                timeframe=row[2],
                last_candle_ts=row[3],
                last_candle_index=row[4],
                active_theta_run_id=row[5],
                active_theta_params=row[6],
                current_phase=row[7],
                phase_direction=row[8],
                cash=row[9],
                updated_at=row[10]
            )

    def get_last_candle_ts(self) -> Optional[pd.Timestamp]:
        """마지막 처리된 캔들 타임스탬프"""
        state = self.load_state()
        if state and state.last_candle_ts:
            return pd.Timestamp(state.last_candle_ts)
        return None

    # ============================================================
    # Utility
    # ============================================================

    def export_all(self) -> Dict[str, Any]:
        """전체 상태 내보내기 (디버깅/백업용)"""
        return {
            "position": self.load_position().to_dict() if self.load_position() else None,
            "state": asdict(self.load_state()) if self.load_state() else None,
            "recent_trades": [asdict(t) for t in self.get_recent_trades(100)],
            "db_path": str(self.db_path),
            "exported_at": datetime.now().isoformat()
        }

    def reset_all(self) -> None:
        """모든 상태 초기화 (주의!)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM positions WHERE symbol = ? AND market = ? AND timeframe = ?",
                          (self.symbol, self.market, self.timeframe))
            cursor.execute("DELETE FROM state WHERE symbol = ? AND market = ? AND timeframe = ?",
                          (self.symbol, self.market, self.timeframe))
            cursor.execute("DELETE FROM trades WHERE symbol = ? AND market = ? AND timeframe = ?",
                          (self.symbol, self.market, self.timeframe))
            conn.commit()


# ============================================================
# Helper Functions
# ============================================================

def get_state_manager(
    symbol: str,
    market: str = "spot",
    timeframe: str = "15m"
) -> StateManager:
    """StateManager 싱글톤 (캐시 가능)"""
    return StateManager(symbol, market, timeframe)


def recover_from_crash(
    symbol: str,
    market: str = "spot",
    timeframe: str = "15m"
) -> Dict[str, Any]:
    """
    크래시 복구 헬퍼

    Returns:
        {
            "has_position": bool,
            "position": PersistedPosition or None,
            "last_candle_ts": Timestamp or None,
            "cash": float,
            "recovery_needed": bool
        }
    """
    state_mgr = get_state_manager(symbol, market, timeframe)

    position = state_mgr.load_position()
    state = state_mgr.load_state()

    has_position = position is not None and position.side != 0
    last_ts = pd.Timestamp(state.last_candle_ts) if state and state.last_candle_ts else None
    cash = state.cash if state else 10000.0

    return {
        "has_position": has_position,
        "position": position,
        "last_candle_ts": last_ts,
        "cash": cash,
        "recovery_needed": has_position or last_ts is not None,
        "state_manager": state_mgr
    }


__all__ = [
    "StateManager",
    "PersistedPosition",
    "PersistedTrade",
    "PersistedState",
    "get_state_manager",
    "recover_from_crash",
]
