from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

@dataclass(frozen=True)
class BacktestCosts:
    fee_bps: float
    slippage_bps: float

@dataclass(frozen=True)
class BacktestConfig:
    # portfolio + exits
    initial_equity: float = 1.0
    max_hold_bars: int = 200
    tp1_frac: float = 0.5
    use_tp2: bool = True

    # WPCN gates (Unknown / No-Trade)
    conf_min: float = 0.55      # min top1 probability
    edge_min: float = 0.75      # min edge_score at signal time
    confirm_bars: int = 1       # top1 must persist for N bars before trading

    # regime safety
    chaos_ret_atr: float = 3.0  # if |ret| > chaos_ret_atr * ATR/close -> CHAOS
    adx_len: int = 14
    adx_trend: float = 20.0     # ADX threshold for trend

@dataclass(frozen=True)
class Theta:
    # box + reclaim contract
    pivot_lr: int
    box_L: int
    m_freeze: int
    atr_len: int
    x_atr: float
    m_bw: float
    N_reclaim: int

    # execution realism knobs
    N_fill: int
    F_min: float

@dataclass(frozen=True)
class RunConfig:
    exchange_id: str
    symbol: str
    timeframe: str
    days: int
    costs: BacktestCosts
    bt: BacktestConfig
    use_tuning: bool
    wf: Dict[str, Any]  # walk-forward settings
    mtf: List[str]      # e.g., ["15m","1h","4h","1d"]
    data_path: Optional[str] = None  # if provided, skip fetching

@dataclass(frozen=True)
class EventSignal:
    t_signal: Any
    side: str    # "long" | "short"
    event: str   # "spring" | "utad"
    entry_price: float
    stop_price: float
    tp1: float
    tp2: float
    meta: Dict[str, Any]
