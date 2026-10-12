"""Grid search and walk-forward testing.

Walk-forward: pick the best settings on a rolling training window (default 3
years), run them on the following test window (default 6 months) that the
search never saw, then roll forward. The stitched test windows are the honest
out-of-sample result.
"""
import itertools

import pandas as pd

from .config import Params
from .metrics import summarize
from .signals import prepare
from . import simulator


def grid(base: Params, space: dict[str, list]) -> list[Params]:
    keys = list(space)
    return [base.with_(**dict(zip(keys, vals)))
            for vals in itertools.product(*(space[k] for k in keys))]


def score(summary: dict) -> float:
    """Default objective: Sharpe, with a penalty when there are too few trades."""
    if not summary or summary.get("trades", 0) < 20:
        return float("-inf")
    return summary["sharpe"]


def search(bars, bench_close, candidates: list[Params], start=None, end=None,
           prepared_cache=None):
    rows = []
    cache = prepared_cache if prepared_cache is not None else {}
    for p in candidates:
        key = _signal_key(p)
        if key not in cache:
            cache[key] = prepare(bars, bench_close, p)
        res = simulator.run(cache[key], p, start, end)
        s = summarize(res.equity, res.trades, res.exposure)
        rows.append({"params": p, "score": score(s), **{k: v for k, v in s.items()
                                                         if not isinstance(v, dict)}})
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def walk_forward(bars, bench_close, candidates: list[Params], train_years=3,
                 test_months=6, start=None, end=None):
    idx = bars["close"].index
    start = pd.Timestamp(start) if start else idx[0]
    end = pd.Timestamp(end) if end else idx[-1]
    cache = {}
    windows, pieces = [], []
    train_start = start
    equity_scale = None
    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)
        if train_end >= end:
            break
        ranked = search(bars, bench_close, candidates, train_start, train_end, cache)
        best = ranked.iloc[0]["params"]
        res = simulator.run(cache[_signal_key(best)], best,
                            train_end + pd.Timedelta(days=1), min(test_end, end))
        if len(res.equity):
            eq = res.equity / res.equity.iloc[0]
            if equity_scale is not None:
                eq = eq * equity_scale
            equity_scale = eq.iloc[-1]
            pieces.append(eq)
            windows.append({"train": f"{train_start.date()}..{train_end.date()}",
                            "test": f"{eq.index[0].date()}..{eq.index[-1].date()}",
                            "best": _diff(best), "train_sharpe": ranked.iloc[0]["score"],
                            "test_return": float(res.equity.iloc[-1] / res.equity.iloc[0] - 1),
                            "test_trades": len(res.trades), "trades": res.trades})
        train_start = train_start + pd.DateOffset(months=test_months)
    oos = pd.concat(pieces) * candidates[0].initial_equity if pieces else pd.Series(dtype=float)
    return oos, windows


def _signal_key(p: Params):
    """Settings that change signals (not exits/sizing) share one prepared dataset."""
    d = p.to_dict()
    for k in ("init_stop_atr", "trail_stop_atr", "partial_target_r", "breakeven_r",
              "time_stop_days", "initial_equity", "risk_per_trade", "max_position_pct",
              "max_positions", "slippage", "commission"):
        d.pop(k)
    return tuple(sorted(d.items()))


def _diff(p: Params) -> dict:
    base = Params().to_dict()
    return {k: v for k, v in p.to_dict().items() if base[k] != v}
