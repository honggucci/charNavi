"""
전략 검증 모듈
- Wyckoff 신호 통계
- RSI/다이버전스 통계
- 신호별 성과 분석
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class SignalStats:
    """신호 통계"""
    total_signals: int
    executed_signals: int
    execution_rate: float  # %
    avg_score: float
    score_distribution: Dict[str, int]  # score 범위별 개수


@dataclass
class TradeStats:
    """거래 통계"""
    total_trades: int
    win_rate: float
    avg_pnl: float
    avg_pnl_pct: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float  # 기대값

    # 청산 유형별
    tp_count: int
    sl_count: int
    timeout_count: int
    liquidation_count: int


@dataclass
class SignalPerformance:
    """신호별 성과"""
    signal_type: str
    trade_count: int
    win_rate: float
    avg_pnl: float
    expectancy: float


class StrategyAnalyzer:
    """
    전략 분석기

    백테스트 결과를 분석하여 전략 개선점 도출
    """

    def analyze(
        self,
        trades_df: pd.DataFrame,
        signals_df: pd.DataFrame
    ) -> Dict:
        """
        전체 분석 실행

        Returns:
            {
                "signal_stats": SignalStats,
                "trade_stats": TradeStats,
                "by_side": {"long": ..., "short": ...},
                "by_reason": {reason: SignalPerformance, ...},
                "by_regime": {regime: TradeStats, ...},
                "recommendations": [str, ...]
            }
        """
        result = {}

        # 신호 통계
        result["signal_stats"] = self._analyze_signals(signals_df)

        # 거래 통계
        result["trade_stats"] = self._analyze_trades(trades_df)

        # 방향별 분석
        result["by_side"] = self._analyze_by_side(trades_df)

        # 청산 유형별 분석
        result["by_exit_reason"] = self._analyze_by_exit_reason(trades_df)

        # 시장 국면별 분석
        if "market_regime" in trades_df.columns:
            result["by_regime"] = self._analyze_by_regime(trades_df)

        # 점수별 성과 분석
        if "signal_score" in trades_df.columns:
            result["by_score"] = self._analyze_by_score(trades_df)

        # 권장사항 생성
        result["recommendations"] = self._generate_recommendations(result)

        return result

    def _analyze_signals(self, signals_df: pd.DataFrame) -> SignalStats:
        """신호 분석"""
        if signals_df is None or len(signals_df) == 0:
            return SignalStats(0, 0, 0, 0, {})

        total = len(signals_df)
        executed = len(signals_df[signals_df.get("executed", False) == True])

        # 점수 분포
        score_dist = {}
        if "score" in signals_df.columns:
            scores = signals_df["score"]
            score_dist = {
                "3-4": len(scores[(scores >= 3) & (scores < 4)]),
                "4-5": len(scores[(scores >= 4) & (scores < 5)]),
                "5-6": len(scores[(scores >= 5) & (scores < 6)]),
                "6+": len(scores[scores >= 6])
            }
            avg_score = scores.mean()
        else:
            avg_score = 0

        return SignalStats(
            total_signals=total,
            executed_signals=executed,
            execution_rate=executed / total * 100 if total > 0 else 0,
            avg_score=avg_score,
            score_distribution=score_dist
        )

    def _analyze_trades(self, trades_df: pd.DataFrame) -> TradeStats:
        """거래 분석"""
        if trades_df is None or len(trades_df) == 0:
            return TradeStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        total = len(trades_df)
        pnl = trades_df["pnl"] if "pnl" in trades_df.columns else pd.Series([0])
        pnl_pct = trades_df["pnl_pct"] if "pnl_pct" in trades_df.columns else pd.Series([0])

        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        win_rate = len(wins) / total * 100 if total > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        # Profit Factor
        total_profit = wins.sum() if len(wins) > 0 else 0
        total_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Expectancy
        expectancy = pnl.mean() if len(pnl) > 0 else 0

        # 청산 유형별 카운트
        exit_reason = trades_df.get("exit_reason", pd.Series(["unknown"] * total))
        tp_count = len(exit_reason[exit_reason == "tp"])
        sl_count = len(exit_reason[exit_reason == "sl"])
        timeout_count = len(exit_reason[exit_reason == "timeout"])
        liq_count = len(exit_reason[exit_reason == "liquidation"])

        return TradeStats(
            total_trades=total,
            win_rate=win_rate,
            avg_pnl=pnl.mean(),
            avg_pnl_pct=pnl_pct.mean() if len(pnl_pct) > 0 else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            tp_count=tp_count,
            sl_count=sl_count,
            timeout_count=timeout_count,
            liquidation_count=liq_count
        )

    def _analyze_by_side(self, trades_df: pd.DataFrame) -> Dict[str, TradeStats]:
        """방향별 분석"""
        result = {}

        if "side" not in trades_df.columns:
            return result

        for side in ["long", "short"]:
            side_df = trades_df[trades_df["side"] == side]
            if len(side_df) > 0:
                result[side] = self._analyze_trades(side_df)

        return result

    def _analyze_by_exit_reason(
        self,
        trades_df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """청산 유형별 분석"""
        result = {}

        if "exit_reason" not in trades_df.columns:
            return result

        for reason in trades_df["exit_reason"].unique():
            reason_df = trades_df[trades_df["exit_reason"] == reason]
            pnl = reason_df["pnl"] if "pnl" in reason_df.columns else pd.Series([0])

            result[reason] = {
                "count": len(reason_df),
                "avg_pnl": pnl.mean(),
                "win_rate": len(pnl[pnl > 0]) / len(reason_df) * 100 if len(reason_df) > 0 else 0
            }

        return result

    def _analyze_by_regime(
        self,
        trades_df: pd.DataFrame
    ) -> Dict[str, TradeStats]:
        """시장 국면별 분석"""
        result = {}

        if "market_regime" not in trades_df.columns:
            return result

        for regime in trades_df["market_regime"].unique():
            regime_df = trades_df[trades_df["market_regime"] == regime]
            if len(regime_df) > 0:
                result[regime] = self._analyze_trades(regime_df)

        return result

    def _analyze_by_score(
        self,
        trades_df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """점수대별 분석"""
        result = {}

        if "signal_score" not in trades_df.columns:
            return result

        score_ranges = [
            ("3.0-3.5", 3.0, 3.5),
            ("3.5-4.0", 3.5, 4.0),
            ("4.0-4.5", 4.0, 4.5),
            ("4.5-5.0", 4.5, 5.0),
            ("5.0+", 5.0, 100)
        ]

        for name, low, high in score_ranges:
            score_df = trades_df[
                (trades_df["signal_score"] >= low) &
                (trades_df["signal_score"] < high)
            ]

            if len(score_df) > 0:
                pnl = score_df["pnl"]
                result[name] = {
                    "count": len(score_df),
                    "win_rate": len(pnl[pnl > 0]) / len(score_df) * 100,
                    "avg_pnl": pnl.mean(),
                    "total_pnl": pnl.sum()
                }

        return result

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        trade_stats = analysis.get("trade_stats")
        if trade_stats:
            # Win rate based recommendation
            if trade_stats.win_rate < 40:
                recommendations.append(
                    f"Win rate is low at {trade_stats.win_rate:.1f}%. "
                    "Consider stronger signal filtering or increase min_score."
                )

            # Stop loss ratio based recommendation
            if trade_stats.sl_count > trade_stats.total_trades * 0.4:
                sl_pct = trade_stats.sl_count / trade_stats.total_trades * 100
                recommendations.append(
                    f"Stop loss ratio is high at {sl_pct:.1f}%. "
                    "Consider wider SL or better entry timing."
                )

            # Take profit ratio based recommendation
            if trade_stats.tp_count < trade_stats.total_trades * 0.2:
                recommendations.append(
                    "TP hit rate is low. "
                    "Consider adjusting TP target or trend following approach."
                )

            # Profit Factor based recommendation
            if trade_stats.profit_factor < 1.0:
                recommendations.append(
                    f"Profit Factor is {trade_stats.profit_factor:.2f} (below 1.0). "
                    "Need to improve RR ratio or strengthen entry conditions."
                )

        # Direction based recommendation
        by_side = analysis.get("by_side", {})
        if "long" in by_side and "short" in by_side:
            long_wr = by_side["long"].win_rate
            short_wr = by_side["short"].win_rate

            if abs(long_wr - short_wr) > 20:
                worse = "SHORT" if short_wr < long_wr else "LONG"
                recommendations.append(
                    f"{worse} position win rate is significantly lower. "
                    f"Review entry conditions for this direction."
                )

        # Market regime based recommendation
        by_regime = analysis.get("by_regime", {})
        for regime, stats in by_regime.items():
            if stats.win_rate < 30:
                recommendations.append(
                    f"Win rate in '{regime}' market is very low at {stats.win_rate:.1f}%. "
                    f"Consider restricting entries in this regime."
                )

        # Score based recommendation
        by_score = analysis.get("by_score", {})
        low_score_losses = 0
        for name, stats in by_score.items():
            if "3.0" in name or "3.5" in name:
                if stats.get("avg_pnl", 0) < 0:
                    low_score_losses += stats.get("count", 0)

        if low_score_losses > 0:
            recommendations.append(
                f"Low score signals (3.0-4.0) produced {low_score_losses} losing trades. "
                "Recommend increasing min_score to 4.0 or higher."
            )

        if not recommendations:
            recommendations.append("Strategy looks good. Consider longer period testing for further optimization.")

        return recommendations

    def print_report(self, analysis: Dict):
        """분석 리포트 출력"""
        print("\n" + "=" * 60)
        print("STRATEGY ANALYSIS REPORT")
        print("=" * 60)

        # 신호 통계
        sig = analysis.get("signal_stats")
        if sig:
            print("\n[Signal Statistics]")
            print(f"  Total Signals: {sig.total_signals}")
            print(f"  Executed: {sig.executed_signals} ({sig.execution_rate:.1f}%)")
            print(f"  Avg Score: {sig.avg_score:.2f}")
            print(f"  Score Distribution: {sig.score_distribution}")

        # 거래 통계
        ts = analysis.get("trade_stats")
        if ts:
            print("\n[Trade Statistics]")
            print(f"  Total Trades: {ts.total_trades}")
            print(f"  Win Rate: {ts.win_rate:.1f}%")
            print(f"  Avg P&L: ${ts.avg_pnl:.2f} ({ts.avg_pnl_pct:.2f}%)")
            print(f"  Avg Win: ${ts.avg_win:.2f}")
            print(f"  Avg Loss: ${ts.avg_loss:.2f}")
            print(f"  Profit Factor: {ts.profit_factor:.2f}")
            print(f"  Expectancy: ${ts.expectancy:.2f}")
            print(f"\n  Exit Types:")
            print(f"    - TP: {ts.tp_count}")
            print(f"    - SL: {ts.sl_count}")
            print(f"    - Timeout: {ts.timeout_count}")
            print(f"    - Liquidation: {ts.liquidation_count}")

        # 방향별 통계
        by_side = analysis.get("by_side", {})
        if by_side:
            print("\n[By Direction]")
            for side, stats in by_side.items():
                print(f"  {side.upper()}: {stats.total_trades} trades, "
                      f"WR={stats.win_rate:.1f}%, "
                      f"Avg=${stats.avg_pnl:.2f}")

        # 점수별 통계
        by_score = analysis.get("by_score", {})
        if by_score:
            print("\n[By Signal Score]")
            for score_range, stats in by_score.items():
                print(f"  {score_range}: {stats['count']} trades, "
                      f"WR={stats['win_rate']:.1f}%, "
                      f"Total=${stats['total_pnl']:.2f}")

        # 권장사항
        recs = analysis.get("recommendations", [])
        if recs:
            print("\n[Recommendations]")
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec}")

        print("\n" + "=" * 60)
