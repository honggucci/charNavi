from __future__ import annotations
import numpy as np
import pandas as pd

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df["high"], df["low"], df["close"])
    return tr.ewm(alpha=1/length, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    rs = up.ewm(alpha=1/length, adjust=False).mean() / (down.ewm(alpha=1/length, adjust=False).mean() + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def stoch_rsi(close: pd.Series, rsi_len: int = 14, stoch_len: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    r = rsi(close, rsi_len)
    r_min = r.rolling(stoch_len, min_periods=stoch_len).min()
    r_max = r.rolling(stoch_len, min_periods=stoch_len).max()
    st = (r - r_min) / (r_max - r_min + 1e-12)
    k = st.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(smooth_d, min_periods=smooth_d).mean()
    return k, d

def zscore(close: pd.Series, length: int = 20) -> pd.Series:
    ma = close.rolling(length, min_periods=length).mean()
    sd = close.rolling(length, min_periods=length).std(ddof=0)
    return (close - ma) / sd.replace(0, np.nan)

def slope(close: pd.Series, length: int = 20) -> pd.Series:
    x = np.arange(length)
    x = x - x.mean()
    denom = (x**2).sum()
    def _sl(y):
        y = np.asarray(y)
        y = y - y.mean()
        return float((x * y).sum() / denom)
    return close.rolling(length, min_periods=length).apply(_sl, raw=False)

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1/length, adjust=False).mean()

    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / (atr_ + 1e-12)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / (atr_ + 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)
    adx_ = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx_


def find_pivot_points(series: pd.Series, lookback: int = 5) -> tuple:
    """Find pivot highs and lows in a series."""
    n = len(series)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    pivot_high_idx = []
    pivot_low_idx = []

    arr = series.values

    for i in range(lookback, n - lookback):
        window = arr[i-lookback:i+lookback+1]
        if arr[i] == np.nanmax(window):
            pivot_highs[i] = arr[i]
            pivot_high_idx.append(i)
        if arr[i] == np.nanmin(window):
            pivot_lows[i] = arr[i]
            pivot_low_idx.append(i)

    return pivot_highs, pivot_lows, pivot_high_idx, pivot_low_idx


def detect_rsi_divergence(df: pd.DataFrame, rsi_len: int = 14, lookback: int = 5) -> pd.DataFrame:
    """
    Detect RSI divergence (bullish and bearish).

    FIXED: 룩어헤드 버그 수정 - 피벗 확정까지 기다린 후 신호 발생
    피벗은 lookback 봉 이후에야 확정되므로, 신호를 lookback만큼 shift함.

    Bullish divergence: Price makes lower low, RSI makes higher low (buy signal)
    Bearish divergence: Price makes higher high, RSI makes lower high (sell signal)
    """
    close = df['close']
    low = df['low']
    high = df['high']
    r = rsi(close, rsi_len)

    n = len(df)
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    div_strength = np.zeros(n)

    # Find pivot points in price and RSI
    _, price_lows, _, price_low_idx = find_pivot_points(low, lookback)
    price_highs, _, price_high_idx, _ = find_pivot_points(high, lookback)
    _, rsi_lows, _, rsi_low_idx = find_pivot_points(r, lookback)
    rsi_highs, _, rsi_high_idx, _ = find_pivot_points(r, lookback)

    low_arr = low.values
    high_arr = high.values
    rsi_arr = r.values

    # Check for bullish divergence (price lower low, RSI higher low)
    for i in range(len(price_low_idx) - 1):
        curr_idx = price_low_idx[i + 1]
        prev_idx = price_low_idx[i]

        if curr_idx - prev_idx < 3 or curr_idx - prev_idx > 50:
            continue

        # Price made lower low
        if low_arr[curr_idx] < low_arr[prev_idx]:
            # Check if RSI made higher low in similar timeframe
            rsi_prev = rsi_arr[prev_idx]
            rsi_curr = rsi_arr[curr_idx]

            # RSI higher low = bullish divergence (더 완화: RSI < 50)
            if rsi_curr > rsi_prev and rsi_curr < 50:  # RSI in lower zone
                bullish_div[curr_idx] = True
                div_strength[curr_idx] = (rsi_curr - rsi_prev) / (rsi_prev + 1e-12)

    # Check for bearish divergence (price higher high, RSI lower high)
    for i in range(len(price_high_idx) - 1):
        curr_idx = price_high_idx[i + 1]
        prev_idx = price_high_idx[i]

        if curr_idx - prev_idx < 3 or curr_idx - prev_idx > 50:
            continue

        # Price made higher high
        if high_arr[curr_idx] > high_arr[prev_idx]:
            # Check if RSI made lower high
            rsi_prev = rsi_arr[prev_idx]
            rsi_curr = rsi_arr[curr_idx]

            # RSI lower high = bearish divergence (더 완화: RSI > 50)
            if rsi_curr < rsi_prev and rsi_curr > 50:  # RSI in upper zone
                bearish_div[curr_idx] = True
                div_strength[curr_idx] = (rsi_prev - rsi_curr) / (rsi_curr + 1e-12)

    # FIXED: 룩어헤드 방지 - 피벗 확정까지 lookback만큼 shift
    # 피벗은 양쪽 lookback 봉을 봐야 확정되므로, 신호를 지연시켜야 현실적
    shift = lookback
    bullish_div = np.roll(bullish_div, shift)
    bullish_div[:shift] = False
    bearish_div = np.roll(bearish_div, shift)
    bearish_div[:shift] = False
    div_strength = np.roll(div_strength, shift)
    div_strength[:shift] = 0.0

    return pd.DataFrame({
        'bullish_div': bullish_div,
        'bearish_div': bearish_div,
        'div_strength': div_strength,
        'rsi': r
    }, index=df.index)


def fibonacci_levels(high: float, low: float) -> dict:
    """Calculate Fibonacci retracement levels."""
    diff = high - low
    return {
        'fib_0': low,
        'fib_236': low + diff * 0.236,
        'fib_382': low + diff * 0.382,
        'fib_500': low + diff * 0.5,
        'fib_618': low + diff * 0.618,
        'fib_786': low + diff * 0.786,
        'fib_100': high,
        # Extensions
        'fib_127': high + diff * 0.272,
        'fib_161': high + diff * 0.618,
    }


def detect_fib_bounce(df: pd.DataFrame, swing_lookback: int = 20, tolerance: float = 0.005) -> pd.DataFrame:
    """
    Detect price bouncing off Fibonacci levels.

    Returns signals when price touches and bounces from key Fib levels (0.382, 0.5, 0.618).
    """
    n = len(df)
    fib_bounce_long = np.zeros(n, dtype=bool)
    fib_bounce_short = np.zeros(n, dtype=bool)
    fib_level = np.full(n, np.nan)

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    for i in range(swing_lookback, n):
        # Find swing high and low in lookback period
        window_high = np.max(high[i-swing_lookback:i])
        window_low = np.min(low[i-swing_lookback:i])

        if window_high - window_low < 1e-12:
            continue

        fibs = fibonacci_levels(window_high, window_low)
        current_close = close[i]
        current_low = low[i]
        current_high = high[i]

        # Check for bounce at key Fib levels (uptrend retracement)
        for level_name, level_price in [('fib_382', fibs['fib_382']),
                                         ('fib_500', fibs['fib_500']),
                                         ('fib_618', fibs['fib_618'])]:
            # Long signal: price dips to Fib level and closes above
            if (current_low <= level_price * (1 + tolerance) and
                current_low >= level_price * (1 - tolerance) and
                current_close > level_price):
                fib_bounce_long[i] = True
                fib_level[i] = level_price
                break

        # Check for rejection at key Fib levels (downtrend retracement)
        for level_name, level_price in [('fib_382', fibs['fib_382']),
                                         ('fib_500', fibs['fib_500']),
                                         ('fib_618', fibs['fib_618'])]:
            # Short signal: price spikes to Fib level and closes below
            if (current_high >= level_price * (1 - tolerance) and
                current_high <= level_price * (1 + tolerance) and
                current_close < level_price):
                fib_bounce_short[i] = True
                fib_level[i] = level_price
                break

    return pd.DataFrame({
        'fib_bounce_long': fib_bounce_long,
        'fib_bounce_short': fib_bounce_short,
        'fib_level': fib_level
    }, index=df.index)
