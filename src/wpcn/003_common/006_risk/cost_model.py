"""
거래 비용 모델
- 수수료 (Maker/Taker)
- 슬리피지
- 펀딩비 (선물)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class CostResult:
    """비용 계산 결과"""
    fee: float           # 수수료
    slippage: float      # 슬리피지
    funding: float       # 펀딩비
    total: float         # 총 비용
    effective_price: float  # 실제 체결가 (슬리피지 반영)


class CostModel(ABC):
    """비용 모델 기본 클래스"""

    @abstractmethod
    def calculate_entry_cost(
        self,
        price: float,
        quantity: float,
        side: Literal["long", "short"],
        order_type: Literal["market", "limit"] = "limit"
    ) -> CostResult:
        """진입 비용 계산"""
        pass

    @abstractmethod
    def calculate_exit_cost(
        self,
        price: float,
        quantity: float,
        side: Literal["long", "short"],
        order_type: Literal["market", "limit"] = "limit"
    ) -> CostResult:
        """청산 비용 계산"""
        pass

    @abstractmethod
    def calculate_funding(
        self,
        position_value: float,
        funding_rate: float,
        hours_held: float
    ) -> float:
        """펀딩비 계산"""
        pass


class BinanceFuturesCost(CostModel):
    """바이낸스 선물 비용 모델"""

    def __init__(
        self,
        maker_fee: float = 0.0002,      # 0.02%
        taker_fee: float = 0.0004,      # 0.04%
        slippage_pct: float = 0.0001,   # 0.01% (시장가 기준)
        avg_funding_rate: float = 0.0001  # 0.01% (8시간마다)
    ):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_pct = slippage_pct
        self.avg_funding_rate = avg_funding_rate

    def calculate_entry_cost(
        self,
        price: float,
        quantity: float,
        side: Literal["long", "short"],
        order_type: Literal["market", "limit"] = "limit"
    ) -> CostResult:
        """진입 비용 계산"""
        notional = price * quantity

        # 수수료
        fee_rate = self.taker_fee if order_type == "market" else self.maker_fee
        fee = notional * fee_rate

        # 슬리피지 (시장가만)
        slippage = 0.0
        effective_price = price
        if order_type == "market":
            slippage_amount = price * self.slippage_pct
            if side == "long":
                effective_price = price + slippage_amount  # 불리하게
            else:
                effective_price = price - slippage_amount
            slippage = abs(effective_price - price) * quantity

        return CostResult(
            fee=fee,
            slippage=slippage,
            funding=0.0,
            total=fee + slippage,
            effective_price=effective_price
        )

    def calculate_exit_cost(
        self,
        price: float,
        quantity: float,
        side: Literal["long", "short"],
        order_type: Literal["market", "limit"] = "limit"
    ) -> CostResult:
        """청산 비용 계산"""
        notional = price * quantity

        # 수수료
        fee_rate = self.taker_fee if order_type == "market" else self.maker_fee
        fee = notional * fee_rate

        # 슬리피지 (시장가만, 청산은 진입 반대 방향)
        slippage = 0.0
        effective_price = price
        if order_type == "market":
            slippage_amount = price * self.slippage_pct
            if side == "long":
                # 롱 청산 = 매도 → 가격 낮게 체결
                effective_price = price - slippage_amount
            else:
                # 숏 청산 = 매수 → 가격 높게 체결
                effective_price = price + slippage_amount
            slippage = abs(effective_price - price) * quantity

        return CostResult(
            fee=fee,
            slippage=slippage,
            funding=0.0,
            total=fee + slippage,
            effective_price=effective_price
        )

    def calculate_funding(
        self,
        position_value: float,
        funding_rate: float = None,
        hours_held: float = 8.0
    ) -> float:
        """
        펀딩비 계산
        - 8시간마다 발생 (UTC 00:00, 08:00, 16:00)
        - 양수 funding_rate: 롱이 숏에게 지불
        - 음수 funding_rate: 숏이 롱에게 지불
        """
        if funding_rate is None:
            funding_rate = self.avg_funding_rate

        # 8시간마다 한번 발생
        funding_periods = hours_held / 8.0
        return position_value * abs(funding_rate) * funding_periods

    def calculate_total_trade_cost(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        side: Literal["long", "short"],
        hours_held: float,
        entry_order: Literal["market", "limit"] = "limit",
        exit_order: Literal["market", "limit"] = "market"  # SL은 보통 시장가
    ) -> dict:
        """전체 거래 비용 계산"""
        entry_cost = self.calculate_entry_cost(entry_price, quantity, side, entry_order)
        exit_cost = self.calculate_exit_cost(exit_price, quantity, side, exit_order)

        avg_position_value = (entry_price + exit_price) / 2 * quantity
        funding = self.calculate_funding(avg_position_value, hours_held=hours_held)

        total_cost = entry_cost.total + exit_cost.total + funding

        # 손익 계산 (비용 반영)
        if side == "long":
            gross_pnl = (exit_cost.effective_price - entry_cost.effective_price) * quantity
        else:
            gross_pnl = (entry_cost.effective_price - exit_cost.effective_price) * quantity

        net_pnl = gross_pnl - entry_cost.fee - exit_cost.fee - funding

        return {
            "entry_cost": entry_cost,
            "exit_cost": exit_cost,
            "funding": funding,
            "total_cost": total_cost,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "cost_pct": total_cost / (entry_price * quantity) * 100
        }


class BinanceSpotCost(CostModel):
    """바이낸스 현물 비용 모델"""

    def __init__(
        self,
        maker_fee: float = 0.001,       # 0.1%
        taker_fee: float = 0.001,       # 0.1%
        slippage_pct: float = 0.0002    # 0.02%
    ):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_pct = slippage_pct

    def calculate_entry_cost(
        self,
        price: float,
        quantity: float,
        side: Literal["long", "short"],
        order_type: Literal["market", "limit"] = "limit"
    ) -> CostResult:
        """진입 비용 (현물은 long만)"""
        notional = price * quantity
        fee_rate = self.taker_fee if order_type == "market" else self.maker_fee
        fee = notional * fee_rate

        slippage = 0.0
        effective_price = price
        if order_type == "market":
            slippage_amount = price * self.slippage_pct
            effective_price = price + slippage_amount
            slippage = slippage_amount * quantity

        return CostResult(
            fee=fee,
            slippage=slippage,
            funding=0.0,
            total=fee + slippage,
            effective_price=effective_price
        )

    def calculate_exit_cost(
        self,
        price: float,
        quantity: float,
        side: Literal["long", "short"],
        order_type: Literal["market", "limit"] = "limit"
    ) -> CostResult:
        """청산 비용"""
        notional = price * quantity
        fee_rate = self.taker_fee if order_type == "market" else self.maker_fee
        fee = notional * fee_rate

        slippage = 0.0
        effective_price = price
        if order_type == "market":
            slippage_amount = price * self.slippage_pct
            effective_price = price - slippage_amount
            slippage = slippage_amount * quantity

        return CostResult(
            fee=fee,
            slippage=slippage,
            funding=0.0,
            total=fee + slippage,
            effective_price=effective_price
        )

    def calculate_funding(
        self,
        position_value: float,
        funding_rate: float,
        hours_held: float
    ) -> float:
        """현물은 펀딩비 없음"""
        return 0.0
