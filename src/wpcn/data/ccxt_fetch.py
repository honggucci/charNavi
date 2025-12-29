from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
import time

import pandas as pd

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def fetch_ohlcv_to_parquet(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    days: int,
    out_path: str,
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
    limit: int = 1500,
    start_date: Optional[str] = None,  # "YYYY-MM-DD" 형식
    end_date: Optional[str] = None,    # "YYYY-MM-DD" 형식
) -> str:
    # reuse the in-memory fetch helper and then save to parquet
    df = fetch_ohlcv_df(
        exchange_id, symbol, timeframe, days,
        api_key=api_key, secret=secret, limit=limit,
        start_date=start_date, end_date=end_date
    )
    df.to_parquet(out_path, index=True)
    return out_path


def fetch_ohlcv_df(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    days: int,
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
    limit: int = 1500,
    start_date: Optional[str] = None,  # "YYYY-MM-DD" 형식
    end_date: Optional[str] = None,    # "YYYY-MM-DD" 형식
) -> pd.DataFrame:
    """Fetch OHLCV from ccxt and return a pandas.DataFrame indexed by datetime.

    This mirrors `fetch_ohlcv_to_parquet` but keeps results in-memory so callers
    can pass the dataframe directly to the backtester without writing parquet.

    Args:
        start_date: 시작 날짜 (YYYY-MM-DD), 지정하면 days 무시
        end_date: 종료 날짜 (YYYY-MM-DD), 기본값은 오늘
    """
    import ccxt  # local import

    exchange_cls = getattr(ccxt, exchange_id)
    params = {"enableRateLimit": True}
    if api_key and secret:
        params.update({"apiKey": api_key, "secret": secret})
    if "options" not in params:
        params["options"] = {}
    if exchange_id in ("binance", "binanceusdm", "binancecoinm"):
        params["options"].setdefault("defaultType", "swap")

    ex = exchange_cls(params)
    ex.load_markets()

    # 시작/종료 날짜 계산
    if start_date and end_date:
        # 특정 기간 지정
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif start_date:
        # 시작 날짜만 지정 → 오늘까지
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = _now_utc()
    else:
        # 기존 방식 (days 사용)
        end = _now_utc()
        start = end - timedelta(days=int(days))

    since = _to_ms(start)
    end_ms = _to_ms(end)

    all_rows = []
    tf_ms = int(ex.parse_timeframe(timeframe) * 1000)

    while True:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        since = last_ts + tf_ms
        if last_ts >= end_ms - tf_ms:
            break
        time.sleep(ex.rateLimit / 1000.0)

    df = pd.DataFrame(all_rows, columns=["time_ms","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time_ms"], unit="ms")
    df = df.drop(columns=["time_ms"]).set_index("time").sort_index()
    df.index.name = "time"
    return df
