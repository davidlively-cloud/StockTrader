"""Indicators on wide frames (rows = dates, columns = symbols).

Every value at row t uses only data up to and including t's close.
"""
import numpy as np
import pandas as pd


def sma(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close.rolling(n, min_periods=n).mean()


def wilder(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder's smoothing (an EMA with alpha = 1/n), NaN until n values exist."""
    out = x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return out


def true_range(high, low, close) -> pd.DataFrame:
    prev = close.shift(1)
    # fmax ignores NaN, so the first bar (no previous close) falls back to high - low
    tr = np.fmax(np.fmax(high - low, (high - prev).abs()), (low - prev).abs())
    return pd.DataFrame(tr, index=close.index, columns=close.columns)


def atr(high, low, close, n: int = 14) -> pd.DataFrame:
    return wilder(true_range(high, low, close), n)


def adx(high, low, close, n: int = 14) -> pd.DataFrame:
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    # keep NaN where the bar itself is missing
    plus_dm = plus_dm.where(high.notna() & high.shift(1).notna())
    minus_dm = minus_dm.where(low.notna() & low.shift(1).notna())
    tr = wilder(true_range(high, low, close), n)
    plus_di = 100 * wilder(plus_dm, n) / tr
    minus_di = 100 * wilder(minus_dm, n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return wilder(dx.replace([np.inf, -np.inf], np.nan), n)


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    gain = wilder(d.clip(lower=0), n)
    loss = wilder((-d).clip(lower=0), n)
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())
