"""
선물 브로커 시뮬레이터
Futures Broker Simulator

레버리지, 마진, 펀딩비, 청산을 반영한 선물 거래 시뮬레이터
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from wpcn.core.types import BacktestCosts
from wpcn.common.cost import apply_costs


@dataclass
class FuturesPosition:
    """선물 포지션"""
    entry_time: Any
    entry_price: float
    size: float  # 계약 수량 (양수=롱, 음수=숏)
    leverage: float
    margin: float  # 사용 마진
    liquidation_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_bar: int = 0


@dataclass
class FuturesTrade:
    """선물 거래 기록"""
    time: Any
    trade_type: str  # 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_TP', 'CLOSE_SL', 'LIQUIDATION', 'CLOSE_TIME'
    price: float
    size: float
    pnl: float
    pnl_pct: float
    leverage: float
    event: str = ''
    margin_used: float = 0.0


class FuturesBroker:
    """
    선물 브로커 시뮬레이터

    특징:
    - 레버리지 거래 지원 (1x ~ 125x)
    - 교차/격리 마진 모드
    - 청산 가격 계산
    - 펀딩비 적용 (8시간마다)
    - 수수료/슬리피지 적용
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        leverage: float = 10.0,
        margin_mode: str = 'isolated',  # 'isolated' or 'cross'
        maintenance_margin_rate: float = 0.005,  # 0.5% 유지마진율
        funding_rate: float = 0.0001,  # 0.01% 펀딩비 (8시간마다)
        max_position_pct: float = 0.5,  # 최대 포지션 비율
        costs: Optional[BacktestCosts] = None
    ):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.margin_mode = margin_mode
        self.maintenance_margin_rate = maintenance_margin_rate
        self.funding_rate = funding_rate
        self.max_position_pct = max_position_pct
        self.costs = costs or BacktestCosts(fee_bps=4.0, slippage_bps=5.0)

        # 상태
        self.balance = initial_balance
        self.position: Optional[FuturesPosition] = None
        self.trades: List[FuturesTrade] = []
        self.equity_history: List[Dict] = []

        # 펀딩비 추적 (8시간 = 32봉 for 15m)
        self.bars_since_funding = 0
        self.funding_interval = 32  # 15분봉 기준 8시간

    def reset(self):
        """상태 초기화"""
        self.balance = self.initial_balance
        self.position = None
        self.trades = []
        self.equity_history = []
        self.bars_since_funding = 0

    def calculate_liquidation_price(
        self,
        entry_price: float,
        size: float,
        margin: float,
        leverage: float
    ) -> float:
        """
        청산 가격 계산

        격리 마진 기준:
        롱: 청산가 = 진입가 × (1 - 1/레버리지 + 유지마진율)
        숏: 청산가 = 진입가 × (1 + 1/레버리지 - 유지마진율)
        """
        if size > 0:  # 롱
            liq_price = entry_price * (1 - 1/leverage + self.maintenance_margin_rate)
        else:  # 숏
            liq_price = entry_price * (1 + 1/leverage - self.maintenance_margin_rate)

        return liq_price

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """미실현 손익 계산"""
        if self.position is None:
            return 0.0

        if self.position.size > 0:  # 롱
            pnl = (current_price - self.position.entry_price) * abs(self.position.size)
        else:  # 숏
            pnl = (self.position.entry_price - current_price) * abs(self.position.size)

        return pnl

    def get_equity(self, current_price: float) -> float:
        """현재 자산 가치"""
        unrealized_pnl = self.calculate_unrealized_pnl(current_price)
        return self.balance + unrealized_pnl

    def open_position(
        self,
        time: Any,
        bar_idx: int,
        price: float,
        side: str,  # 'long' or 'short'
        position_pct: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        event: str = ''
    ) -> bool:
        """
        포지션 오픈

        Args:
            time: 시간
            bar_idx: 바 인덱스
            price: 진입 가격
            side: 'long' or 'short'
            position_pct: 잔고 대비 포지션 비율 (0~1)
            stop_loss: 손절가
            take_profit: 익절가
            event: 이벤트 설명

        Returns:
            성공 여부
        """
        if self.position is not None:
            return False  # 이미 포지션 있음

        # 비용 적용
        fill_price = apply_costs(price, self.costs)

        # 마진 계산
        position_pct = min(position_pct, self.max_position_pct)
        margin = self.balance * position_pct

        if margin < 10:  # 최소 마진
            return False

        # 포지션 크기 (레버리지 적용)
        notional_value = margin * self.leverage
        size = notional_value / fill_price

        if side == 'short':
            size = -size

        # 청산가 계산
        liq_price = self.calculate_liquidation_price(fill_price, size, margin, self.leverage)

        # 포지션 생성
        self.position = FuturesPosition(
            entry_time=time,
            entry_price=fill_price,
            size=size,
            leverage=self.leverage,
            margin=margin,
            liquidation_price=liq_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_bar=bar_idx
        )

        # 마진 차감
        self.balance -= margin

        # 거래 기록
        trade_type = 'OPEN_LONG' if side == 'long' else 'OPEN_SHORT'
        self.trades.append(FuturesTrade(
            time=time,
            trade_type=trade_type,
            price=fill_price,
            size=size,
            pnl=0.0,
            pnl_pct=0.0,
            leverage=self.leverage,
            event=event,
            margin_used=margin
        ))

        return True

    def close_position(
        self,
        time: Any,
        price: float,
        reason: str,  # 'TP', 'SL', 'LIQUIDATION', 'TIME', 'SIGNAL'
        event: str = ''
    ) -> Optional[float]:
        """
        포지션 청산

        Returns:
            실현 PnL (없으면 None)
        """
        if self.position is None:
            return None

        # 비용 적용
        fill_price = apply_costs(price, self.costs)

        # PnL 계산
        if self.position.size > 0:  # 롱
            pnl = (fill_price - self.position.entry_price) * abs(self.position.size)
        else:  # 숏
            pnl = (self.position.entry_price - fill_price) * abs(self.position.size)

        pnl_pct = pnl / self.position.margin * 100

        # 청산 시 마진 + PnL 반환
        if reason == 'LIQUIDATION':
            # 청산 시 마진 전액 손실
            self.balance += 0
            pnl = -self.position.margin
            pnl_pct = -100.0
        else:
            self.balance += self.position.margin + pnl

        # 거래 기록
        trade_type = f'CLOSE_{reason}'
        self.trades.append(FuturesTrade(
            time=time,
            trade_type=trade_type,
            price=fill_price,
            size=-self.position.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            leverage=self.position.leverage,
            event=event,
            margin_used=self.position.margin
        ))

        self.position = None
        return pnl

    def apply_funding_fee(self, time: Any, current_price: float):
        """펀딩비 적용 (8시간마다)"""
        if self.position is None:
            return

        # 펀딩비 = 포지션 가치 × 펀딩비율
        position_value = abs(self.position.size) * current_price

        if self.position.size > 0:  # 롱
            # 롱 포지션은 양수 펀딩비일 때 지불
            funding_fee = position_value * self.funding_rate
        else:  # 숏
            # 숏 포지션은 양수 펀딩비일 때 수취
            funding_fee = -position_value * self.funding_rate

        self.balance -= funding_fee

    def check_liquidation(self, current_price: float) -> bool:
        """청산 여부 확인"""
        if self.position is None:
            return False

        if self.position.size > 0:  # 롱
            return current_price <= self.position.liquidation_price
        else:  # 숏
            return current_price >= self.position.liquidation_price

    def check_stop_loss(self, low: float, high: float) -> bool:
        """손절 여부 확인"""
        if self.position is None or self.position.stop_loss is None:
            return False

        if self.position.size > 0:  # 롱
            return low <= self.position.stop_loss
        else:  # 숏
            return high >= self.position.stop_loss

    def check_take_profit(self, low: float, high: float) -> bool:
        """익절 여부 확인"""
        if self.position is None or self.position.take_profit is None:
            return False

        if self.position.size > 0:  # 롱
            return high >= self.position.take_profit
        else:  # 숏
            return low <= self.position.take_profit

    def process_bar(
        self,
        time: Any,
        bar_idx: int,
        open_price: float,
        high: float,
        low: float,
        close: float,
        max_hold_bars: int = 100
    ) -> Optional[str]:
        """
        바 처리 (청산/손절/익절 체크)

        Returns:
            발생한 이벤트 ('LIQUIDATION', 'SL', 'TP', 'TIME', None)
        """
        # 펀딩비 처리
        self.bars_since_funding += 1
        if self.bars_since_funding >= self.funding_interval:
            self.apply_funding_fee(time, close)
            self.bars_since_funding = 0

        if self.position is None:
            # 자산 기록
            self.equity_history.append({
                'time': time,
                'equity': self.balance,
                'balance': self.balance,
                'unrealized_pnl': 0.0,
                'position_size': 0.0,
                'position_side': 0
            })
            return None

        # 진입 바에서는 청산 검사 스킵
        if bar_idx <= self.position.entry_bar:
            equity = self.get_equity(close)
            self.equity_history.append({
                'time': time,
                'equity': equity,
                'balance': self.balance,
                'unrealized_pnl': self.calculate_unrealized_pnl(close),
                'position_size': abs(self.position.size),
                'position_side': 1 if self.position.size > 0 else -1
            })
            return None

        # 1. 청산 체크 (가장 먼저)
        if self.check_liquidation(low if self.position.size > 0 else high):
            self.close_position(time, self.position.liquidation_price, 'LIQUIDATION')
            self.equity_history.append({
                'time': time,
                'equity': self.balance,
                'balance': self.balance,
                'unrealized_pnl': 0.0,
                'position_size': 0.0,
                'position_side': 0
            })
            return 'LIQUIDATION'

        # 2. 손절 체크
        if self.check_stop_loss(low, high):
            sl_price = self.position.stop_loss
            self.close_position(time, sl_price, 'SL')
            self.equity_history.append({
                'time': time,
                'equity': self.balance,
                'balance': self.balance,
                'unrealized_pnl': 0.0,
                'position_size': 0.0,
                'position_side': 0
            })
            return 'SL'

        # 3. 익절 체크
        if self.check_take_profit(low, high):
            tp_price = self.position.take_profit
            self.close_position(time, tp_price, 'TP')
            self.equity_history.append({
                'time': time,
                'equity': self.balance,
                'balance': self.balance,
                'unrealized_pnl': 0.0,
                'position_size': 0.0,
                'position_side': 0
            })
            return 'TP'

        # 4. 시간 청산 체크
        if bar_idx - self.position.entry_bar >= max_hold_bars:
            self.close_position(time, close, 'TIME')
            self.equity_history.append({
                'time': time,
                'equity': self.balance,
                'balance': self.balance,
                'unrealized_pnl': 0.0,
                'position_size': 0.0,
                'position_side': 0
            })
            return 'TIME'

        # 자산 기록
        equity = self.get_equity(close)
        self.equity_history.append({
            'time': time,
            'equity': equity,
            'balance': self.balance,
            'unrealized_pnl': self.calculate_unrealized_pnl(close),
            'position_size': abs(self.position.size),
            'position_side': 1 if self.position.size > 0 else -1
        })

        return None

    def get_equity_df(self) -> pd.DataFrame:
        """자산 히스토리 DataFrame 반환"""
        if not self.equity_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.equity_history)
        df = df.set_index('time')
        df['ret'] = df['equity'].pct_change().fillna(0.0)
        return df

    def get_trades_df(self) -> pd.DataFrame:
        """거래 히스토리 DataFrame 반환"""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([{
            'time': t.time,
            'type': t.trade_type,
            'price': t.price,
            'size': t.size,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'leverage': t.leverage,
            'event': t.event,
            'margin': t.margin_used
        } for t in self.trades])

    def get_statistics(self) -> Dict[str, Any]:
        """거래 통계"""
        if not self.trades:
            return {}

        trades_df = self.get_trades_df()
        equity_df = self.get_equity_df()

        # 청산 거래만 필터
        close_trades = trades_df[trades_df['type'].str.startswith('CLOSE')]

        if len(close_trades) == 0:
            return {
                'total_trades': 0,
                'total_return': (self.balance - self.initial_balance) / self.initial_balance * 100
            }

        wins = close_trades[close_trades['pnl'] > 0]
        losses = close_trades[close_trades['pnl'] <= 0]

        win_rate = len(wins) / len(close_trades) * 100 if len(close_trades) > 0 else 0
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl_pct'].mean()) if len(losses) > 0 else 0
        profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf')

        # MDD 계산
        if len(equity_df) > 0:
            cummax = equity_df['equity'].cummax()
            drawdown = (equity_df['equity'] - cummax) / cummax * 100
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0

        # 청산 횟수
        liquidations = len(close_trades[close_trades['type'] == 'CLOSE_LIQUIDATION'])

        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'total_trades': len(close_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'liquidations': liquidations,
            'leverage': self.leverage
        }
