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

    # 인덱스 이름 확인 및 설정
    left_reset = left.reset_index()
    left_reset.columns = ["t"]  # 명시적으로 컬럼명 설정

    right_reset = right.reset_index()
    if right_reset.columns[0] != "t":
        right_reset.rename(columns={right_reset.columns[0]: "t"}, inplace=True)

    merged = pd.merge_asof(
        left_reset,
        right_reset,
        on="t",
        direction="backward"
    ).set_index("t")
    return merged
