from __future__ import annotations
import pandas as pd

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["open"].resample(rule, label="right", closed="right").first()
    h = df["high"].resample(rule, label="right", closed="right").max()
    l = df["low"].resample(rule, label="right", closed="right").min()
    c = df["close"].resample(rule, label="right", closed="right").last()
    v = df["volume"].resample(rule, label="right", closed="right").sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()
    return out

def merge_asof_back(left_index: pd.DatetimeIndex, right_df: pd.DataFrame) -> pd.DataFrame:
    right = right_df.sort_index()
    left = pd.DataFrame(index=left_index).sort_index()
    merged = pd.merge_asof(
        left.reset_index().rename(columns={"index":"t"}),
        right.reset_index().rename(columns={right.index.name or "index":"t"}),
        on="t",
        direction="backward"
    ).set_index("t")
    return merged
