import numpy as np
import pandas as pd

from trendbt.scanner import ScanParams, evaluate, volume_fraction


def bars(closes, volume=10_000):
    c = np.asarray(closes, float)
    t = pd.date_range("2026-10-12 13:30", periods=len(c), freq="5min", tz="UTC")
    return pd.DataFrame({"open": c - 0.05, "high": c + 0.1, "low": c - 0.1, "close": c,
                         "volume": float(volume), "vwap": c}, index=t)


P = ScanParams()


def test_volume_curve_is_monotonic_and_complete():
    assert volume_fraction(0) == 0 and volume_fraction(390) == 1
    assert volume_fraction(60) < volume_fraction(61) < volume_fraction(390)


def test_steady_uptrend_on_heavy_volume_matches_with_a_plan():
    b = bars(np.linspace(103, 106, 18), volume=200_000)       # 90 minutes in
    r = evaluate(b, prev_close=100, avg_volume=4_000_000, daily_atr=3.0, minutes=90, p=P)
    assert r["match"], r["checks"]
    assert r["stop"] < r["price"] - 0.25 * 3.0 + 1e-9
    assert r["target"] == r["price"] + 2 * (r["price"] - r["stop"])
    assert 0 < r["shares"] * r["price"] <= 0.25 * 25_000 + 1e-6


def test_fails_on_normal_volume_downtrend_or_chase():
    up = np.linspace(103, 106, 18)
    quiet = evaluate(bars(up, 20_000), 100, 4_000_000, 3.0, 90, P)
    assert not quiet["checks"]["in_play"] and not quiet["match"]
    down = evaluate(bars(up[::-1], 200_000), 100, 4_000_000, 3.0, 90, P)
    assert not down["checks"]["above_vwap"] and not down["match"]
    spike = evaluate(bars(list(np.full(17, 103.0)) + [110], 200_000), 100, 4_000_000, 3.0, 90, P)
    assert not spike["checks"]["not_chased"]


def test_too_few_bars_returns_none():
    assert evaluate(bars([101, 102, 103]), 100, 1e6, 2.0, 15, P) is None


def test_market_filter_reads_spy_against_its_vwap():
    from trendbt.scanner import market_ok
    up, down = bars(np.linspace(100, 102, 12)), bars(np.linspace(102, 100, 12))
    asof = up.index[-1] + pd.Timedelta(minutes=5)
    assert market_ok(up, asof) and not market_ok(down, asof)
    assert market_ok(None, asof)
