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
) -> str:
    import ccxt  # local import

    exchange_cls = getattr(ccxt, exchange_id)
    params = {"enableRateLimit": True}
    if api_key and secret:
        params.update({"apiKey": api_key, "secret": secret})
    # For some exchanges you may need: options.defaultType = 'swap' or 'future'.
    if "options" not in params:
        params["options"] = {}
    if exchange_id in ("binance", "binanceusdm", "binancecoinm"):
        # USDT-M futures recommended: binanceusdm.
        params["options"].setdefault("defaultType", "swap")

    ex = exchange_cls(params)
    ex.load_markets()

    end = _now_utc()
    start = end - timedelta(days=int(days))
    since = _to_ms(start)
    end_ms = _to_ms(end)

    # reuse the in-memory fetch helper and then save to parquet
    df = fetch_ohlcv_df(exchange_id, symbol, timeframe, days, api_key=api_key, secret=secret, limit=limit)
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
) -> pd.DataFrame:
    """Fetch OHLCV from ccxt and return a pandas.DataFrame indexed by datetime.

    This mirrors `fetch_ohlcv_to_parquet` but keeps results in-memory so callers
    can pass the dataframe directly to the backtester without writing parquet.
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
