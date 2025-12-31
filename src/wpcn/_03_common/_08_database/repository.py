"""
데이터 접근 레이어 (Repository Pattern)
"""

from typing import List, Optional
from datetime import datetime
from .connection import execute_query, execute_non_query, get_cursor
from .models import Trade, Signal, OptimizationResult


class TradeRepository:
    """거래 기록 저장소"""

    @staticmethod
    def insert(trade: Trade) -> int:
        """거래 기록 삽입"""
        query = """
        INSERT INTO trades (
            symbol, side, entry_price, exit_price, quantity,
            tp_price, sl_price, pnl, pnl_pct, fee,
            entry_time, exit_time, exit_reason, signal_score,
            market_regime, timeframe, strategy_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            trade.symbol, trade.side, trade.entry_price, trade.exit_price,
            trade.quantity, trade.tp_price, trade.sl_price, trade.pnl,
            trade.pnl_pct, trade.fee, trade.entry_time, trade.exit_time,
            trade.exit_reason, trade.signal_score, trade.market_regime,
            trade.timeframe, trade.strategy_version
        )
        return execute_non_query(query, params)

    @staticmethod
    def update_exit(trade_id: int, exit_price: float, exit_time: datetime,
                    exit_reason: str, pnl: float, pnl_pct: float) -> int:
        """거래 종료 업데이트"""
        query = """
        UPDATE trades SET
            exit_price = ?, exit_time = ?, exit_reason = ?,
            pnl = ?, pnl_pct = ?
        WHERE id = ?
        """
        return execute_non_query(query, (exit_price, exit_time, exit_reason, pnl, pnl_pct, trade_id))

    @staticmethod
    def get_by_symbol(symbol: str, limit: int = 100) -> List[dict]:
        """심볼별 거래 조회"""
        query = """
        SELECT TOP (?) * FROM trades
        WHERE symbol = ?
        ORDER BY entry_time DESC
        """
        return execute_query(query, (limit, symbol))

    @staticmethod
    def get_recent(limit: int = 50) -> List[dict]:
        """최근 거래 조회"""
        query = f"SELECT TOP ({limit}) * FROM trades ORDER BY entry_time DESC"
        return execute_query(query)

    @staticmethod
    def get_stats(symbol: str = None, start_date: datetime = None) -> dict:
        """거래 통계"""
        where_clauses = ["exit_price IS NOT NULL"]
        params = []

        if symbol:
            where_clauses.append("symbol = ?")
            params.append(symbol)
        if start_date:
            where_clauses.append("entry_time >= ?")
            params.append(start_date)

        where_sql = " AND ".join(where_clauses)

        query = f"""
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
            SUM(pnl) as total_pnl,
            AVG(pnl_pct) as avg_pnl_pct,
            MAX(pnl_pct) as max_pnl_pct,
            MIN(pnl_pct) as min_pnl_pct,
            SUM(fee) as total_fees
        FROM trades
        WHERE {where_sql}
        """
        result = execute_query(query, tuple(params) if params else None)
        return result[0] if result else {}


class SignalRepository:
    """신호 로그 저장소"""

    @staticmethod
    def insert(signal: Signal) -> int:
        """신호 로그 삽입"""
        import json
        query = """
        INSERT INTO signals (
            symbol, timeframe, timestamp, action, score,
            long_score, short_score, reason, tf_alignment, executed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            signal.symbol, signal.timeframe, signal.timestamp, signal.action,
            signal.score, signal.long_score, signal.short_score,
            json.dumps(signal.reason), signal.tf_alignment, signal.executed
        )
        return execute_non_query(query, params)

    @staticmethod
    def get_recent(symbol: str = None, limit: int = 100) -> List[dict]:
        """최근 신호 조회"""
        if symbol:
            query = f"SELECT TOP ({limit}) * FROM signals WHERE symbol = ? ORDER BY timestamp DESC"
            return execute_query(query, (symbol,))
        else:
            query = f"SELECT TOP ({limit}) * FROM signals ORDER BY timestamp DESC"
            return execute_query(query)


class OptimizationRepository:
    """최적화 결과 저장소"""

    @staticmethod
    def insert(result: OptimizationResult) -> int:
        """최적화 결과 삽입"""
        import json
        query = """
        INSERT INTO optimization_results (
            week, symbol, timeframe, params, backtest_result,
            sharpe_ratio, total_return, max_drawdown, win_rate, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            result.week, result.symbol, result.timeframe,
            json.dumps(result.params), json.dumps(result.backtest_result),
            result.sharpe_ratio, result.total_return, result.max_drawdown,
            result.win_rate, result.created_at
        )
        return execute_non_query(query, params)

    @staticmethod
    def get_latest(symbol: str, timeframe: str) -> Optional[dict]:
        """최신 최적화 결과 조회"""
        query = """
        SELECT TOP 1 * FROM optimization_results
        WHERE symbol = ? AND timeframe = ?
        ORDER BY created_at DESC
        """
        result = execute_query(query, (symbol, timeframe))
        return result[0] if result else None

    @staticmethod
    def get_by_week(week: str) -> List[dict]:
        """주차별 최적화 결과 조회"""
        query = "SELECT * FROM optimization_results WHERE week = ?"
        return execute_query(query, (week,))
