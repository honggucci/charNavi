"""
백테스트 설정
Backtest configurations

=== 2026-01-01 수정사항 ===
1. 비용 현실화: 0.4% → BTC 0.105%, ALT 0.20%
2. Gate 상향: conf 0.50→0.65, edge 0.60→0.70
3. reclaim_hold_bars 추가: 휩쏘 방지
4. Scalping/Navigation 프로파일 분리
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class BacktestSettings:
    """백테스트 공통 설정"""
    # 초기 자본
    initial_equity: float = 1.0

    # 비용 설정 (현실화)
    # - BTC Spot: fee=7.5bps, slip=3bps (총 10.5bps)
    # - ALT Spot: fee=10bps, slip=10bps (총 20bps)
    # - Futures: fee=4bps, slip=5bps (총 9bps)
    fee_bps: float = 7.5       # 0.075% 수수료 (BNB 할인)
    slippage_bps: float = 5.0  # 0.05% 슬리피지 (현실적)

    # 시간 청산
    max_hold_bars: int = 192   # Navigation: 48시간 (15m*192)

    # TP 설정
    tp1_frac: float = 0.5      # TP1에서 50% 청산
    use_tp2: bool = True

    # 신호 필터 (상향 조정)
    # 철학: "안 들어가는 날이 많아져야 계좌가 산다"
    conf_min: float = 0.65     # 0.50 → 0.65 (쓰레기 신호 필터링)
    edge_min: float = 0.70     # 0.60 → 0.70 (엣지 없으면 진입 금지)

    # 휩쏘 방지
    confirm_bars: int = 2      # 1 → 2 (신호 확정 대기)
    reclaim_hold_bars: int = 2 # NEW: reclaim 후 박스 내부 유지 확인


@dataclass
class ScalpingSettings(BacktestSettings):
    """단타 백테스트 설정"""
    # 시간 청산 (짧게)
    max_hold_bars: int = 30    # 7.5시간 (15m*30)

    # 신호 필터 (약간 완화)
    conf_min: float = 0.60     # 단타: 약간 완화
    edge_min: float = 0.65

    # 빠른 진입
    confirm_bars: int = 1
    reclaim_hold_bars: int = 1


@dataclass
class FuturesBacktestSettings(BacktestSettings):
    """선물 백테스트 설정"""
    # 비용 (선물)
    fee_bps: float = 4.0       # 0.04% 수수료 (Maker)
    slippage_bps: float = 5.0  # 0.05% 슬리피지

    # 레버리지
    leverage: float = 5.0      # 10 → 5 (보수적)
    margin_mode: str = 'isolated'

    # 유지 증거금
    maintenance_margin_rate: float = 0.005  # 0.5%

    # 펀딩비
    funding_rate: float = 0.0001  # 0.01% per 8 hours
    funding_interval_bars: int = 96  # 5분봉 기준 8시간

    # 청산 방지 버퍼
    liquidation_buffer_pct: float = 0.50  # 50% 버퍼


# 기본 설정 (프로파일별)
DEFAULT_BACKTEST_SETTINGS = BacktestSettings()
DEFAULT_SCALPING_SETTINGS = ScalpingSettings()
DEFAULT_FUTURES_SETTINGS = FuturesBacktestSettings()
