"""Universe filter, trend filter and entry signals.

All outputs are evaluated at a day's close; the simulator acts on them at the
next day's open, so nothing here can use information the trader would not have.
"""
from dataclasses import dataclass

import pandas as pd

from . import indicators as ind
from .config import Params


@dataclass
class Prepared:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    atr: pd.DataFrame
    exit_ma: pd.DataFrame | None
    entry: pd.DataFrame        # bool: entry signal at this close
    rank: pd.DataFrame         # priority when signals exceed open slots (higher first)


def regime(bench_close: pd.Series, p: Params) -> pd.Series:
    if not p.use_regime_filter:
        return pd.Series(True, index=bench_close.index)
    return bench_close > ind.sma(bench_close.to_frame(), p.slow_ma).iloc[:, 0]


def prepare(bars: dict[str, pd.DataFrame], bench_close: pd.Series, p: Params) -> Prepared:
    o, h, l, c, v = (bars[k] for k in ("open", "high", "low", "close", "volume"))

    dollar_vol = (c * v).rolling(20, min_periods=20).mean()
    universe = (c > p.min_price) & (dollar_vol > p.min_dollar_volume)

    fast = ind.sma(c, p.fast_ma)
    slow = ind.sma(c, p.slow_ma)
    adx = ind.adx(h, l, c, p.adx_period)
    ret = c / c.shift(p.rs_lookback) - 1
    # relative strength percentile among stocks that pass the universe filter today
    rs_pct = ret.where(universe).rank(axis=1, pct=True)

    trend = (
        (c > fast)
        & (fast > slow)
        & (fast > fast.shift(p.slope_lookback))
        & (adx > p.min_adx)
        & (rs_pct >= p.min_rs_pct)
    )

    ok_regime = regime(bench_close.reindex(c.index), p)
    base = universe & trend
    base = base.mul(ok_regime.fillna(False).astype(bool), axis=0)

    breakout = c > c.shift(1).rolling(p.breakout_days, min_periods=p.breakout_days).max()

    pb_ma = ind.sma(c, p.pullback_ma)
    rsi = ind.rsi(c, p.rsi_period)
    dipped = (l.shift(1) <= pb_ma.shift(1)) | (rsi.shift(1) < p.rsi_oversold)
    pullback = dipped & (c > c.shift(1))

    if p.entry == "breakout":
        sig = base & breakout
    elif p.entry == "pullback":
        sig = base & pullback
    elif p.entry == "any":
        sig = base & (breakout | pullback)
    elif p.entry == "momentum":
        # top N by 6-month return among trend-qualified stocks, only on the
        # last trading day of each week (orders go in for Monday's open)
        top = ret.where(base).rank(axis=1, ascending=False) <= p.momentum_top_n
        idx = c.index
        week = pd.Series(idx.to_period("W"), index=idx)
        last_of_week = week != week.shift(-1)
        sig = top.mul(last_of_week, axis=0)
    else:
        raise ValueError(f"unknown entry type {p.entry!r}")

    exit_ma = ind.sma(c, p.trend_exit_ma) if p.trend_exit_ma else None
    return Prepared(
        open=o, high=h, low=l, close=c,
        atr=ind.atr(h, l, c, p.atr_period),
        exit_ma=exit_ma,
        entry=sig.fillna(False).astype(bool),
        rank=ret,
    )
