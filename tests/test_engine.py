import numpy as np
import pandas as pd
import pytest

from trendbt import indicators as ind, simulator
from trendbt.config import Params
from trendbt.signals import Prepared, prepare
from trendbt.synthetic import random_market
from trendbt.walkforward import grid, walk_forward


def frame(values, dates=None, sym="X"):
    dates = dates if dates is not None else pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame({sym: np.asarray(values, float)}, index=dates)


def hand_prepared(o, h, l, c, signal_days, atr=1.0, exit_ma=None):
    """A one-stock dataset with a fixed ATR and signals on chosen bars."""
    O, H, L, C = (frame(x) for x in (o, h, l, c))
    sig = pd.DataFrame(False, index=C.index, columns=C.columns)
    sig.iloc[signal_days, 0] = True
    return Prepared(open=O, high=H, low=L, close=C,
                    atr=pd.DataFrame(atr, index=C.index, columns=C.columns),
                    exit_ma=frame(exit_ma) if exit_ma is not None else None,
                    entry=sig, rank=C * 0 + 1)


NO_EXTRAS = dict(partial_target_r=0, breakeven_r=0, time_stop_days=0, slippage=0)


# ---------- indicators ----------

def test_true_range_and_atr():
    h, l, c = frame([10, 12, 11]), frame([9, 10, 8]), frame([9.5, 11, 9])
    tr = ind.true_range(h, l, c)["X"].tolist()
    assert tr == [1, 2.5, 3]            # 2nd: 12-9.5, 3rd: 11-8
    a = ind.atr(h, l, c, 2)["X"]
    assert np.isnan(a.iloc[0]) and a.iloc[1] == pytest.approx(2.5 * 0.5 + 1 * 0.5)


def test_rsi_bounds_and_direction():
    up = frame(np.arange(1, 40))
    assert ind.rsi(up, 14)["X"].dropna().eq(100).all()
    zig = frame([10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10])
    r = ind.rsi(zig, 2)["X"].dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_adx_high_in_steady_trend():
    n = 120
    c = frame(np.linspace(10, 50, n))
    a = ind.adx(c + 0.2, c - 0.2, c, 14)["X"]
    assert a.iloc[-1] > 50


# ---------- simulator ----------

def test_entry_next_open_and_sizing():
    # signal on bar 0 -> buy at bar 1's open; risk 1% of 25k with a 2-ATR stop of $2
    o = h = l = c = [100, 100, 100, 100]
    res = simulator.run(hand_prepared(o, h, l, c, [0]), Params(**NO_EXTRAS))
    t = res.trades.iloc[0]
    assert t.entry_date == res.equity.index[1] and t.entry_price == 100
    # 1% risk = $250 / $2 stop = 125 shares, but the 10% position cap (25 shares) wins
    assert t.shares == 25 and t.pnl == 0 and t.reason == "end_of_test"
    p = Params(**NO_EXTRAS, max_position_pct=1.0)
    res = simulator.run(hand_prepared(o, h, l, c, [0]), p)
    assert res.trades.iloc[0].shares == 125


def test_stop_and_gap_fills():
    p = Params(**NO_EXTRAS, max_position_pct=1.0)
    # entry 100 at bar 1, stop 98; bar 2 trades down to 97 -> filled at the stop
    res = simulator.run(hand_prepared([100, 100, 99, 99], [100, 100, 99, 99],
                                      [100, 100, 97, 99], [100, 100, 98.5, 99], [0]), p)
    t = res.trades.iloc[0]
    assert t.reason == "stop" and t.exit_price == 98 and t.pnl == pytest.approx(-250)
    # gap open at 95, below the 98 stop -> filled at the open, a bigger loss
    res = simulator.run(hand_prepared([100, 100, 95, 95], [100, 100, 96, 95],
                                      [100, 100, 94, 95], [100, 100, 95, 95], [0]), p)
    t = res.trades.iloc[0]
    assert t.reason == "stop_gap" and t.exit_price == 95 and t.r_multiple == pytest.approx(-2.5)


def test_trailing_stop_ratchets_up():
    # rally to 110 then fall; trail = highest close (110) - 3*ATR(1) = 107
    c = [100, 100, 105, 110, 108, 106, 106]
    o = [100, 100, 104, 109, 109, 107.5, 106]
    h = [x + 0.5 for x in c]
    l = [100, 100, 104, 109, 107.5, 105, 105]
    res = simulator.run(hand_prepared(o, h, l, c, [0]), Params(**NO_EXTRAS))
    t = res.trades.iloc[0]
    assert t.reason == "stop" and t.exit_price == 107


def test_partial_target_and_breakeven():
    p = Params(**{**NO_EXTRAS, "partial_target_r": 2.0, "breakeven_r": 1.0},
               max_position_pct=1.0)
    # entry 100, R = 2; target 104 hit on bar 2 (sell half), then stop raised to
    # breakeven 100 and hit on bar 4
    o = [100, 100, 101, 103, 101]
    h = [100, 100, 104.5, 103.5, 101]
    l = [100, 100, 100.5, 102, 99]
    c = [100, 100, 103, 102.5, 100]
    res = simulator.run(hand_prepared(o, h, l, c, [0]), p)
    t = res.trades.iloc[0]
    assert t.reason == "stop" and t.exit_price == 100
    assert t.pnl == pytest.approx(62 * 4 + 63 * 0)   # 125 shares: 62 sold at 104, rest at 100


def test_time_stop_and_trend_break():
    flat = [100] * 15
    p = Params(**{**NO_EXTRAS, "time_stop_days": 5})
    res = simulator.run(hand_prepared(flat, flat, flat, flat, [0]), p)
    assert res.trades.iloc[0].reason == "time_stop" and res.trades.iloc[0].days == 5
    c = [100, 100, 101, 99, 99]
    res = simulator.run(hand_prepared(c, c, c, c, [0], exit_ma=[100] * 5),
                        Params(**NO_EXTRAS, trend_exit_ma=1))
    t = res.trades.iloc[0]
    assert t.reason == "trend_break" and t.exit_date == res.equity.index[4]


def test_max_positions_respected():
    bars, spy = random_market(n_symbols=40, seed=1)
    p = Params(entry="any", max_positions=3, max_position_pct=0.5)
    res = simulator.run(prepare(bars, spy, p), p)
    tr = res.trades
    for d in res.equity.index[::20]:
        held = ((tr.entry_date <= d) & (tr.exit_date > d)).sum()
        assert held <= 3


def test_no_lookahead():
    """Changing prices after a date must not change any trade entered before it."""
    bars, spy = random_market(n_symbols=30, seed=2, end="2021-12-31")
    p = Params(entry="any")
    cut = pd.Timestamp("2020-06-30")
    full = simulator.run(prepare(bars, spy, p), p, end=cut)
    shocked = {k: v.copy() for k, v in bars.items()}
    for k in ("open", "high", "low", "close"):
        shocked[k].loc[shocked[k].index > cut] *= 3.0
    spy2 = spy.copy()
    spy2[spy2.index > cut] *= 0.2
    alt = simulator.run(prepare(shocked, spy2, p), p, end=cut)
    pd.testing.assert_series_equal(full.equity, alt.equity)
    pd.testing.assert_frame_equal(full.trades, alt.trades)


def test_equity_accounting_matches_trades():
    bars, spy = random_market(n_symbols=30, seed=3)
    p = Params(entry="breakout")
    res = simulator.run(prepare(bars, spy, p), p)
    assert res.trades.pnl.sum() == pytest.approx(
        res.equity.iloc[-1] - p.initial_equity, rel=1e-3)


def test_walk_forward_runs():
    bars, spy = random_market(n_symbols=20, seed=4, end="2021-12-31")
    cands = grid(Params(), {"trail_stop_atr": [2.0, 3.0]})
    oos, windows = walk_forward(bars, spy, cands)
    assert len(windows) >= 2 and oos.index.is_monotonic_increasing
