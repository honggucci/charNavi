from __future__ import annotations
from wpcn.core.types import BacktestCosts

def apply_costs(price: float, costs: BacktestCosts) -> float:
    bps = (costs.fee_bps + costs.slippage_bps) / 10000.0
    return price * (1.0 + bps)
