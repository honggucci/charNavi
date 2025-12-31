"""
백테스트 설정
Backtest configurations
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class BacktestSettings:
    """백테스트 공통 설정"""
    # 초기 자본
    initial_equity: float = 1.0

    # 비용 설정
    fee_bps: float = 4.0      # 0.04% 수수료
    slippage_bps: float = 5.0  # 0.05% 슬리피지

    # 시간 청산
    max_hold_bars: int = 30

    # TP 설정
    tp1_frac: float = 0.5  # TP1에서 50% 청산
    use_tp2: bool = True

    # 신호 필터
    conf_min: float = 0.50
    edge_min: float = 0.60


@dataclass
class FuturesBacktestSettings(BacktestSettings):
    """선물 백테스트 설정"""
    # 레버리지
    leverage: float = 10.0
    margin_mode: str = 'isolated'

    # 유지 증거금
    maintenance_margin_rate: float = 0.005  # 0.5%

    # 펀딩비
    funding_rate: float = 0.0001  # 0.01% per 8 hours
    funding_interval_bars: int = 96  # 5분봉 기준 8시간


# 기본 설정
DEFAULT_BACKTEST_SETTINGS = BacktestSettings()
DEFAULT_FUTURES_SETTINGS = FuturesBacktestSettings()
