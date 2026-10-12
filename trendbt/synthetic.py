"""Synthetic price data for testing the engine. Results on it mean nothing
about real markets; it only proves the code runs and the rules fire."""
import numpy as np
import pandas as pd


def random_market(n_symbols=60, start="2016-01-01", end="2025-12-31", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    # market factor plus stock-specific drift that changes regime now and then
    mkt = rng.normal(0.0003, 0.010, n)
    out = {f: {} for f in ("open", "high", "low", "close", "volume")}
    for k in range(n_symbols):
        drift = np.repeat(rng.normal(0.0004, 0.0015, n // 250 + 1), 250)[:n]
        r = drift + 1.1 * mkt + rng.normal(0, 0.018, n)
        close = 30 * np.exp(np.cumsum(r))
        gap = rng.normal(0, 0.006, n)
        open_ = np.r_[close[0], close[:-1]] * np.exp(gap)
        hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.008, n)))
        lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.008, n)))
        vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
        sym = f"S{k:03d}"
        for f, v in zip(("open", "high", "low", "close", "volume"), (open_, hi, lo, close, vol)):
            out[f][sym] = v
    bars = {f: pd.DataFrame(v, index=dates) for f, v in out.items()}
    spy = pd.Series(300 * np.exp(np.cumsum(mkt)), index=dates, name="SPY")
    return bars, spy
