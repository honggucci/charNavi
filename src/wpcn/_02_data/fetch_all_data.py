"""
전체 데이터 수집 스크립트
- 현물/선물 병렬 처리
- 월별 parquet 저장
- bronze 구조로 저장
"""

import os
import time
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd

# 설정
SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "DOGE/USDT",
           "LINK/USDT", "ZEC/USDT", "ICP/USDT", "FIL/USDT", "TRX/USDT"]

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d", "1w"]

# 각 타임프레임별 시작 연도 (너무 오래된 데이터는 없을 수 있음)
START_YEAR = 2019

BASE_DIR = Path(__file__).parent.parent.parent.parent  # wpcn-backtester-cli-noflask
DATA_DIR = BASE_DIR / "data" / "bronze"


def get_exchange(market_type: str):
    """거래소 객체 생성"""
    import ccxt

    if market_type == "spot":
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}
        })
    else:  # futures
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })

    exchange.load_markets()
    return exchange


def fetch_ohlcv_for_period(exchange, symbol: str, timeframe: str,
                           start_date: datetime, end_date: datetime,
                           limit: int = 1500) -> pd.DataFrame:
    """특정 기간의 OHLCV 데이터 수집"""

    since = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    tf_ms = int(exchange.parse_timeframe(timeframe) * 1000)

    all_rows = []

    while since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not ohlcv:
                break

            all_rows.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            since = last_ts + tf_ms

            if last_ts >= end_ms - tf_ms:
                break

            time.sleep(exchange.rateLimit / 1000.0)

        except Exception as e:
            print(f"      [ERROR] {e}")
            time.sleep(5)
            continue

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["time_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["time_ms"], unit="ms")
    df = df.drop(columns=["time_ms"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return df


def save_by_month(df: pd.DataFrame, output_dir: Path):
    """월별로 분리하여 저장"""
    if df.empty:
        return 0

    df["_year"] = df["timestamp"].dt.year
    df["_month"] = df["timestamp"].dt.month

    saved_count = 0

    for (year, month), group in df.groupby(["_year", "_month"]):
        group = group.drop(columns=["_year", "_month"])

        year_dir = output_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        output_file = year_dir / f"{month:02d}.parquet"

        # 기존 파일이 있으면 병합
        if output_file.exists():
            existing = pd.read_parquet(output_file)
            group = pd.concat([existing, group])
            group = group.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        group.to_parquet(output_file, index=False)
        saved_count += len(group)

    return saved_count


def fetch_symbol_data(exchange, symbol: str, timeframe: str, market_type: str):
    """심볼 하나의 전체 데이터 수집"""

    symbol_clean = symbol.replace("/", "-")
    output_dir = DATA_DIR / "binance" / market_type / symbol_clean / timeframe

    # 시작/끝 날짜
    start = datetime(START_YEAR, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)

    print(f"    [{market_type}] {symbol} {timeframe}: {start.year} ~ {end.year}")

    try:
        df = fetch_ohlcv_for_period(exchange, symbol, timeframe, start, end)

        if df.empty:
            print(f"      → 데이터 없음")
            return

        saved = save_by_month(df, output_dir)
        print(f"      → {saved:,}행 저장")

    except Exception as e:
        print(f"      → [ERROR] {e}")


def fetch_market_data(market_type: str):
    """특정 마켓(spot/futures)의 모든 데이터 수집"""

    print(f"\n{'='*60}")
    print(f"[{market_type.upper()}] 데이터 수집 시작")
    print(f"{'='*60}")

    exchange = get_exchange(market_type)

    for symbol in SYMBOLS:
        print(f"\n  {symbol}:")

        for timeframe in TIMEFRAMES:
            fetch_symbol_data(exchange, symbol, timeframe, market_type)
            time.sleep(1)  # 심볼/타임프레임 간 잠시 대기

    print(f"\n[{market_type.upper()}] 완료!")


def main():
    print("="*60)
    print("전체 데이터 수집 시작")
    print(f"코인: {len(SYMBOLS)}개")
    print(f"타임프레임: {TIMEFRAMES}")
    print(f"마켓: spot, futures (병렬)")
    print("="*60)

    start_time = time.time()

    # 현물/선물 병렬 실행
    spot_thread = threading.Thread(target=fetch_market_data, args=("spot",))
    futures_thread = threading.Thread(target=fetch_market_data, args=("futures",))

    spot_thread.start()
    futures_thread.start()

    spot_thread.join()
    futures_thread.join()

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print("\n" + "="*60)
    print(f"전체 완료! 소요 시간: {hours}시간 {minutes}분")
    print("="*60)


if __name__ == "__main__":
    main()
