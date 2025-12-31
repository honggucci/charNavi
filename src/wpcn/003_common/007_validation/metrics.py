"""
성과 지표 계산 모듈
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- 기타 통계
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """성과 지표 모음"""
    # 수익 관련
    total_return: float          # 총 수익률 (%)
    annual_return: float         # 연간 수익률 (%)
    monthly_return_avg: float    # 월평균 수익률 (%)

    # 리스크 관련
    max_drawdown: float          # 최대 낙폭 (%)
    volatility: float            # 변동성 (연간화)
    downside_volatility: float   # 하방 변동성

    # 위험 조정 수익률
    sharpe_ratio: float          # Sharpe Ratio
    sortino_ratio: float         # Sortino Ratio
    calmar_ratio: float          # Calmar Ratio

    # 거래 통계
    total_trades: int            # 총 거래 수
    win_rate: float              # 승률 (%)
    profit_factor: float         # Profit Factor
    avg_win: float               # 평균 수익 (%)
    avg_loss: float              # 평균 손실 (%)
    max_consecutive_losses: int  # 최대 연속 손실

    # 통계적 유의성
    t_statistic: float           # t-통계량
    p_value: float               # p-value


def calculate_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate: float = 0.02,  # 연간 무위험 수익률 2%
    periods_per_year: int = 252    # 거래일 기준
) -> PerformanceMetrics:
    """
    성과 지표 계산

    Args:
        equity_curve: 자산 곡선 (시계열)
        trades: 거래 기록 DataFrame (pnl_pct 컬럼 필요)
        risk_free_rate: 무위험 수익률
        periods_per_year: 연간 기간 수
    """
    # 수익률 계산
    returns = equity_curve.pct_change().dropna()

    if len(returns) == 0:
        return _empty_metrics()

    # 총 수익률
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

    # 연간화
    n_periods = len(returns)
    years = n_periods / periods_per_year
    annual_return = ((1 + total_return / 100) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0

    # 월평균 수익률
    monthly_return_avg = total_return / max(years * 12, 1)

    # 최대 낙폭
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = abs(drawdowns.min()) * 100

    # 변동성 (연간화)
    volatility = returns.std() * np.sqrt(periods_per_year) * 100

    # 하방 변동성 (음수 수익률만)
    negative_returns = returns[returns < 0]
    downside_volatility = negative_returns.std() * np.sqrt(periods_per_year) * 100 if len(negative_returns) > 0 else 0

    # Sharpe Ratio
    excess_return = annual_return - risk_free_rate * 100
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0

    # Sortino Ratio
    sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0

    # Calmar Ratio
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # 거래 통계
    if len(trades) > 0 and 'pnl_pct' in trades.columns:
        total_trades = len(trades)
        wins = trades[trades['pnl_pct'] > 0]
        losses = trades[trades['pnl_pct'] <= 0]

        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl_pct'].mean()) if len(losses) > 0 else 0

        # Profit Factor
        gross_profit = wins['pnl_pct'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # 최대 연속 손실
        max_consecutive_losses = _calculate_max_consecutive_losses(trades['pnl_pct'])

    else:
        total_trades = 0
        win_rate = 0
        profit_factor = 0
        avg_win = 0
        avg_loss = 0
        max_consecutive_losses = 0

    # 통계적 유의성 (t-test)
    t_stat, p_val = _calculate_significance(returns)

    return PerformanceMetrics(
        total_return=round(total_return, 2),
        annual_return=round(annual_return, 2),
        monthly_return_avg=round(monthly_return_avg, 2),
        max_drawdown=round(max_drawdown, 2),
        volatility=round(volatility, 2),
        downside_volatility=round(downside_volatility, 2),
        sharpe_ratio=round(sharpe_ratio, 3),
        sortino_ratio=round(sortino_ratio, 3),
        calmar_ratio=round(calmar_ratio, 3),
        total_trades=total_trades,
        win_rate=round(win_rate, 2),
        profit_factor=round(profit_factor, 2),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        max_consecutive_losses=max_consecutive_losses,
        t_statistic=round(t_stat, 3),
        p_value=round(p_val, 4)
    )


def _calculate_max_consecutive_losses(pnl_series: pd.Series) -> int:
    """최대 연속 손실 계산"""
    max_streak = 0
    current_streak = 0

    for pnl in pnl_series:
        if pnl <= 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def _calculate_significance(returns: pd.Series) -> tuple:
    """t-test로 통계적 유의성 계산"""
    from scipy import stats

    if len(returns) < 2:
        return 0.0, 1.0

    # H0: 평균 수익률 = 0
    t_stat, p_value = stats.ttest_1samp(returns, 0)

    return t_stat, p_value


def _empty_metrics() -> PerformanceMetrics:
    """빈 메트릭스"""
    return PerformanceMetrics(
        total_return=0, annual_return=0, monthly_return_avg=0,
        max_drawdown=0, volatility=0, downside_volatility=0,
        sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
        total_trades=0, win_rate=0, profit_factor=0,
        avg_win=0, avg_loss=0, max_consecutive_losses=0,
        t_statistic=0, p_value=1.0
    )


def compare_metrics(metrics_list: List[PerformanceMetrics], names: List[str]) -> pd.DataFrame:
    """여러 전략/기간의 메트릭스 비교"""
    data = []
    for m, name in zip(metrics_list, names):
        data.append({
            "Strategy": name,
            "Return (%)": m.total_return,
            "Annual (%)": m.annual_return,
            "MDD (%)": m.max_drawdown,
            "Sharpe": m.sharpe_ratio,
            "Sortino": m.sortino_ratio,
            "Win Rate (%)": m.win_rate,
            "Profit Factor": m.profit_factor,
            "Trades": m.total_trades,
            "p-value": m.p_value
        })

    return pd.DataFrame(data)
