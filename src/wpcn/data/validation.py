from __future__ import annotations
import pandas as pd

REQUIRED_COLS = ["open","high","low","close","volume"]

def validate_ohlcv_df(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df.index must be sorted ascending")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df[REQUIRED_COLS].isna().any().any():
        raise ValueError("df contains NaNs in required OHLCV columns")
