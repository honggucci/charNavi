from __future__ import annotations
import pandas as pd
from .validation import validate_ohlcv_df

def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "time" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("time")
    df.index = pd.to_datetime(df.index, utc=False)
    df = df.sort_index()
    validate_ohlcv_df(df)
    return df

def save_parquet(df: pd.DataFrame, path: str) -> None:
    out = df.copy()
    out.index.name = "time"
    out.to_parquet(path, index=True)
