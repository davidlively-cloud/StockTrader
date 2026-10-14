import json
import re
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trendbt import webapp
from trendbt.scanner import ScanParams, evaluate

PAGE = Path(webapp.__file__).with_name("web") / "index.html"


def bars(closes, volume=10_000, vwap_lag=0.0):
    c = np.asarray(closes, float)
    t = pd.date_range("2026-10-12 13:30", periods=len(c), freq="5min", tz="UTC")
    return pd.DataFrame({"open": c - 0.05, "high": c + 0.1, "low": c - 0.1, "close": c,
                         "volume": float(volume), "vwap": c - vwap_lag}, index=t)


CASES = [   # (closes, volume, prev_close, avg_volume, atr, minutes)
    (np.linspace(103, 106, 18), 200_000, 100, 4_000_000, 3.0, 90),     # clean uptrend
    (np.linspace(106, 103, 18), 200_000, 100, 4_000_000, 3.0, 90),     # downtrend
    (np.linspace(101, 102, 30), 20_000, 100, 4_000_000, 3.0, 150),     # quiet
    (np.r_[np.linspace(104, 112, 20), [112.5] * 4], 300_000, 100, 4_000_000, 2.0, 120),  # stretched
    (np.linspace(4.2, 4.6, 24), 500_000, 4.0, 3_000_000, 0.3, 120),    # a $4 stock
]
SETTINGS = [ScanParams(), ScanParams(max_price=1000),
            ScanParams(max_price=1000, min_rel_volume=1, min_change=0.01, max_atr_above_vwap=2,
                       min_stop_atr=0.3, target_r=3, equity=50_000)]


def test_measure_keeps_what_the_page_needs():
    r = webapp.measure(bars(*CASES[0][:2]), *CASES[0][2:], p=ScanParams())
    assert {"price", "change", "rel_volume", "vwap", "atr", "avg_volume", "stop_base",
            "vwap_rising", "ema_trend"} <= set(r)
    assert webapp.measure(bars([100, 101]), 100, 1e6, 2.0, 10, ScanParams()) is None


def _page_evaluate(rows, params):
    """Run the page's own evaluate() in node on measured rows."""
    js = re.search(r"function evaluate\(r, p\) \{.*?\n\}", PAGE.read_text(), re.S).group(0)
    prog = js + "\nconst [rows, ps] = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n" \
        "console.log(JSON.stringify(ps.map(p => rows.map(r => { const e = evaluate(r, p);" \
        " return [e.match, e.shares, e.stop, e.target, e.failed]; }))));"
    out = subprocess.run(["node", "-e", prog], input=json.dumps([rows, params]),
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_page_applies_settings_exactly_like_the_scanner():
    rows = [webapp.measure(bars(c, v), pc, av, atr, m, ScanParams()) for c, v, pc, av, atr, m in CASES]
    page = _page_evaluate(rows, [asdict(p) for p in SETTINGS])
    for p, got in zip(SETTINGS, page):
        for (c, v, pc, av, atr, m), (match, shares, stop, target, failed) in zip(CASES, got):
            want = evaluate(bars(c, v), pc, av, atr, m, p)
            assert match == want["match"]
            assert sorted(failed) == sorted(k for k, ok in want["checks"].items() if not ok)
            assert shares == want["shares"]
            assert stop == pytest.approx(want["stop"]) and target == pytest.approx(want["target"])


def test_follow_status_rules():
    wl = {"UP": {"prev_close": 9, "pm_high": 10, "pm_low": 9.5, "gap": 0.1},
          "DN": {"prev_close": 9, "pm_high": 10, "pm_low": 9.5, "gap": 0.1}}
    out = {r["symbol"]: r["status"] for r in webapp.Store.follow(None, wl, {
        "UP": bars(np.linspace(9.8, 10.6, 12), vwap_lag=0.3),
        "DN": bars(np.linspace(9.8, 9.2, 12))})}
    assert out == {"UP": "breaking out", "DN": "failed"}


def test_universe_settings_and_lookback_time():
    p = webapp.params_from({"max_price": ["20"], "min_dollar_volume": ["1000000"], "top": ["3"]})
    assert (p.max_price, p.min_dollar_volume, p.top) == (20, 1e6, ScanParams().top)
    asof, live = webapp.resolve_asof({"at": ["2026-10-13 11:02"]})
    assert (f"{asof:%H:%M}", live) == ("11:00", False)
    with pytest.raises(ValueError):
        webapp.resolve_asof({"at": ["2026-10-13 08:00"]})
