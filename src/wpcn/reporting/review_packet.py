from __future__ import annotations

def build_review_prompt() -> str:
    return (
        "You are a strict quant reviewer. Review this WPCN Spring/UTAD backtest run.\n"
        "1) Check for lookahead/leakage risks (signals using future data, resampling errors).\n"
        "2) Check whether entry/exit simulation is overly optimistic. Suggest conservative adjustments.\n"
        "3) Identify overfitting signs in walk-forward and parameter stability.\n"
        "4) Recommend next validation experiments and stress tests.\n"
        "Return: bullet list with severity levels and concrete fixes.\n"
    )
