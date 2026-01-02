from __future__ import annotations
from wpcn.core.types import BacktestCosts

def apply_costs(price: float, costs: BacktestCosts, is_buy: bool = True) -> float:
    """
    거래 비용 적용

    매수(is_buy=True): 비싸게 삼 → price * (1 + bps)
    매도(is_buy=False): 싸게 팖 → price * (1 - bps)

    Args:
        price: 원래 가격
        costs: 비용 설정
        is_buy: True=매수, False=매도

    Returns:
        비용 적용된 체결 가격
    """
    bps = (costs.fee_bps + costs.slippage_bps) / 10000.0
    if is_buy:
        return price * (1.0 + bps)  # 매수: 비싸게
    else:
        return price * (1.0 - bps)  # 매도: 싸게
