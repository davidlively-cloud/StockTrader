"""Web dashboard for the intraday scanner: market tone, trend alerts, the
premarket watchlist, steady climbers (while SPY is weak) and a chart per stock,
with the scan settings editable in the page.

  python -m trendbt.webapp                  # http://127.0.0.1:8050
  python -m trendbt.webapp --port 8080 --cache data/scan
  python -m trendbt.webapp --snapshot view.html --at "2026-10-14 14:00"  # save one view as a page

The server does the slow part (downloading 5-minute bars for every stock in the
price/volume universe and measuring each one with scanner.evaluate) and sends
the measurements to the page. The page applies the thresholds, so changing a
setting re-filters instantly; only the universe settings (price range and
dollar volume) make the server download again.

Data is Alpaca's free SIP feed, 15 minutes late, the same as the scanner.
"""
import argparse
import json
import threading
from dataclasses import asdict, replace
from datetime import time as dtime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from . import data, scanner
from .scanner import ET, OPEN, CLOSE, SIP_DELAY, ScanParams

PAGE = Path(__file__).with_name("web") / "index.html"
HEAD = ('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">')
BAR = pd.Timedelta(minutes=5)
# Universe settings: changing these needs a new download
UNIVERSE_KEYS = ("min_price", "max_price", "min_dollar_volume")
# The page's own defaults. David (2026-10-19): Trend Desk shows stocks at $1 or less;
# the hourly alert scans keep the scanner's own limit.
WEB_PARAMS = replace(ScanParams(), max_price=1.0)
# Charts start here so they show the premarket run-up (small caps trade little before 7)
CHART_START = dtime(7, 0)
# Market tone. Alpaca's free feed has no index quotes, so these are the funds that
# track each index. SPY alone gates the alerts, as in the scanner.
INDEX_FUNDS = {"SPY": "S&P 500", "DIA": "Dow Jones Industrial Average", "ONEQ": "Nasdaq Composite"}


def _floor5(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.floor("5min")


def trading_days(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    cal = data._get(f"{data.ALPACA_TRADING}/calendar",
                    {"start": f"{start:%Y-%m-%d}", "end": f"{end:%Y-%m-%d}"},
                    data._alpaca_headers())
    return [pd.Timestamp(d["date"], tz=ET) for d in cal]


def latest_asof(now: pd.Timestamp | None = None) -> tuple[pd.Timestamp, bool]:
    """The moment to show: 15 minutes ago from 7:05 am to the close (premarket
    included), else the last close. Returns (asof, live)."""
    now = now or pd.Timestamp.now(tz=ET)
    days = trading_days(now - pd.Timedelta(days=10), now)
    today = now.normalize()
    if days and days[-1] == today:
        start = pd.Timestamp.combine(today.date(), CHART_START).tz_localize(ET)
        close = pd.Timestamp.combine(today.date(), CLOSE).tz_localize(ET)
        asof = _floor5(now - SIP_DELAY)
        if start + BAR <= asof < close:
            return asof, True
        if asof >= close:
            return close, False
        days = days[:-1]
    last = days[-1]
    return pd.Timestamp.combine(last.date(), CLOSE).tz_localize(ET), False


def measure(bars: pd.DataFrame, prev_close: float, avg_volume: float, atr: float,
            minutes: float, p: ScanParams) -> dict | None:
    """What the page needs to apply the scan's thresholds itself: scanner.evaluate
    run with the price and volume limits wide open, plus the stop before its
    minimum-distance rule. The page's evaluate() turns this back into the scan."""
    loose = replace(p, min_price=0, max_price=1e9, min_dollar_volume=0)
    r = scanner.evaluate(bars, prev_close, avg_volume, atr, minutes, loose)
    if not r:
        return None
    low = float(bars["low"].iloc[-p.stop_lookback_bars:].min())
    return {"price": r["price"], "change": r["change"], "rel_volume": r["rel_volume"],
            "vwap": r["vwap"], "atr": float(atr), "avg_volume": float(avg_volume),
            "prev_close": float(prev_close),
            # under the higher of VWAP and the recent low, as in evaluate
            "stop_base": max(r["vwap"], low) - 0.05 * float(atr),
            "vwap_rising": bool(r["checks"]["vwap_rising"]), "ema_trend": bool(r["checks"]["ema_trend"]),
            "volume": float(bars["volume"].sum())}


def _num(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    if isinstance(x, (np.floating, float)):
        return round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


class Store:
    """Downloads and caches bars; one lock so two page loads don't both download."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.lock = threading.Lock()
        self.symbols: dict[str, list[str]] = {}        # day -> tradable stocks
        self.views: dict[tuple, dict] = {}              # (asof, universe) -> scan view
        self.daily: dict[tuple, pd.DataFrame] = {}      # (symbol, day) -> daily bars

    def stats(self, day: pd.Timestamp) -> pd.DataFrame:
        key = f"{day:%Y-%m-%d}"
        if key not in self.symbols:
            self.symbols[key] = scanner.tradable_stocks()
        return scanner.daily_stats(self.symbols[key], day, self.cache_dir)

    def view(self, asof: pd.Timestamp, p: ScanParams) -> dict:
        key = (asof, tuple(getattr(p, k) for k in UNIVERSE_KEYS))
        with self.lock:
            if key not in self.views:
                if len(self.views) > 6:
                    self.views.pop(next(iter(self.views)))
                self.views[key] = self._build(asof, p)
            return self.views[key]

    def _build(self, asof: pd.Timestamp, p: ScanParams) -> dict:
        day = asof.normalize()
        session_open = pd.Timestamp.combine(day.date(), OPEN).tz_localize(ET)
        minutes = (asof - session_open).total_seconds() / 60
        stats = self.stats(day)
        stats = stats[(stats["prev_close"] >= p.min_price) & (stats["prev_close"] < p.max_price * 1.3)
                      & (stats["prev_close"] * stats["avg_volume"] >= p.min_dollar_volume)]
        # the scanner's premarket list uses its own price limit; keep the page's universe
        wl = {s: lv for s, lv in self.watchlist(asof).items()
              if p.min_price <= lv["prev_close"] < p.max_price * 1.3}
        syms = sorted(set(stats.index) | set(INDEX_FUNDS) | set(wl))
        chart_start = pd.Timestamp.combine(day.date(), CHART_START).tz_localize(ET)
        pm_start = pd.Timestamp.combine(day.date(), scanner.PREMARKET_OPEN).tz_localize(ET)
        allbars = scanner.intraday_bars(syms, pm_start, asof)
        allbars = {s: b[b.index + BAR <= asof] for s, b in allbars.items()}
        # the scan reads the regular session only; the premarket part is for charts
        bars = {s: b[b.index >= session_open] for s, b in allbars.items()}
        pmbars = {s: b[(b.index >= chart_start) & (b.index < session_open)] for s, b in allbars.items()}
        pm_volume = self.pm_volume(allbars, stats, session_open)
        day_volume = {s: float(b["volume"].sum()) for s, b in allbars.items()}   # from 4:00 am
        rows = []
        for sym, st in stats.iterrows():
            b = bars.get(sym)
            r = measure(b, st.prev_close, st.avg_volume, st.atr, minutes, p) if b is not None else None
            if r:
                rows.append({"symbol": sym, **r})
        market = {"indexes": [], "premarket": minutes <= 0}
        for sym, name in INDEX_FUNDS.items():
            b = bars.get(sym)
            if b is None or not len(b):
                b = pmbars.get(sym)            # before the open: premarket trading so far
            if b is None or not len(b):
                continue
            last = float(b["close"].iloc[-1])
            vw = float((b["vwap"] * b["volume"]).sum() / b["volume"].sum())
            prev = self.prev_close(sym, day)
            market["indexes"].append({"symbol": sym, "name": name, "last": last, "vwap": vw,
                                      "above_vwap": last > vw, "prev_close": prev,
                                      "change": last / prev - 1 if prev else None})
            if sym == "SPY":
                market.update(spy=last, spy_vwap=vw, spy_above_vwap=last > vw,
                              spy_change=last / prev - 1 if prev else None)
        climbers = []
        if "spy" in market and not market["spy_above_vwap"]:
            climbers = self.climbers(bars, stats, minutes, p)
        return {"asof": asof, "minutes": minutes, "rows": rows, "market": market,
                "bars": bars, "pmbars": pmbars, "stats": stats, "watchlist": wl, "climbers": climbers,
                "follow": [r for r in self.follow(asof, wl, bars, pmbars) if r["last"] < p.max_price], "pm_volume": pm_volume, "day_volume": day_volume}

    @staticmethod
    def pm_volume(allbars: dict, stats: pd.DataFrame, session_open: pd.Timestamp) -> dict:
        """Premarket volume (4:00-9:30) as a multiple of each stock's average full-day volume."""
        out = {}
        for sym, b in allbars.items():
            if sym not in stats.index or not stats.loc[sym, "avg_volume"] > 0:
                continue
            x = float(b.loc[b.index < session_open, "volume"].sum()) / stats.loc[sym, "avg_volume"]
            if x >= 0.05:
                out[sym] = round(x, 3)
        return out

    @staticmethod
    def climbers(bars: dict, stats: pd.DataFrame, minutes: float, p: ScanParams) -> list[dict]:
        """Steady climbers by the scanner's fixed rules (scanner.CLIMBER, is_climber),
        within the page's price and volume universe. Only asked for while SPY is weak."""
        cp = replace(scanner.CLIMBER, **{k: getattr(p, k) for k in UNIVERSE_KEYS})
        out = []
        for sym, st in stats.iterrows():
            b = bars.get(sym)
            r = scanner.evaluate(b, st.prev_close, st.avg_volume, st.atr, minutes, cp) if b is not None else None
            if r and scanner.is_climber(r):
                out.append({"symbol": sym, **{k: r[k] for k in (
                    "price", "change", "rel_volume", "vwap", "stop", "target", "risk_pct", "shares",
                    "climb_fit", "above_vwap_share", "score")}})
        return sorted(out, key=lambda r: -r["score"])[:cp.top]

    def prev_close(self, sym: str, day: pd.Timestamp) -> float | None:
        d = self.daily_bars(sym, day)
        d = d[d.index < day.tz_localize(None)]
        return float(d["close"].iloc[-1]) if len(d) else None

    def watchlist(self, asof: pd.Timestamp) -> dict:
        """This day's premarket names: the scanner's saved watchlist, or, when the
        premarket scan didn't run that day, the 9:30 premarket scan run now
        (before the open, the premarket scan as of now, not saved)."""
        path = self.cache_dir / f"watchlist_{asof:%Y-%m-%d}.json"
        if path.exists():
            return json.loads(path.read_text())
        open_ = pd.Timestamp.combine(asof.date(), OPEN).tz_localize(ET)
        if asof < open_:
            hits, _ = scanner.premarket(asof, cache_dir=self.cache_dir)
            return {r["symbol"]: {k: float(r[k]) for k in ("prev_close", "pm_high", "pm_low", "gap")}
                    for _, r in hits.iterrows()}
        hits, info = scanner.premarket(open_, cache_dir=self.cache_dir)
        scanner.save_watchlist(hits, info, self.cache_dir)     # once per day is enough
        return {r["symbol"]: {k: float(r[k]) for k in ("prev_close", "pm_high", "pm_low", "gap")}
                for _, r in hits.iterrows()}

    @staticmethod
    def follow(asof, wl: dict, bars: dict, pmbars: dict | None = None) -> list[dict]:
        """Status of each premarket name, by the same rules as scanner.follow.
        Before the open, the premarket price so far with status "premarket"."""
        out = []
        for sym, lv in wl.items():
            b = bars.get(sym)
            if b is None or not len(b):
                pb = (pmbars or {}).get(sym)
                if pb is not None and len(pb):
                    last = float(pb["close"].iloc[-1])
                    out.append({"symbol": sym, "last": last, "change": last / lv["prev_close"] - 1,
                                "from_open": None, "vwap": None, "gap": lv.get("gap"),
                                "pm_high": lv["pm_high"], "pm_low": lv["pm_low"],
                                "day_high": None, "status": "premarket"})
                continue
            last = float(b["close"].iloc[-1])
            vwap = float((b["vwap"] * b["volume"]).sum() / b["volume"].sum())
            if last < lv["pm_low"]:
                status = "failed"
            elif last > lv["pm_high"] and last > vwap:
                status = "breaking out"
            elif last > vwap:
                status = "holding"
            else:
                status = "fading"
            out.append({"symbol": sym, "last": last, "change": last / lv["prev_close"] - 1,
                        "from_open": last / float(b["open"].iloc[0]) - 1, "vwap": vwap,
                        "gap": lv.get("gap"), "pm_high": lv["pm_high"], "pm_low": lv["pm_low"],
                        "day_high": float(b["high"].max()), "status": status})
        return sorted(out, key=lambda r: -(r["from_open"] if r["from_open"] is not None else r["change"]))

    def daily_bars(self, sym: str, day: pd.Timestamp) -> pd.DataFrame:
        key = (sym, f"{day:%Y-%m-%d}")
        if key not in self.daily:
            # sessions before `day` only: the free plan refuses SIP data from the last 15 minutes
            start = (day - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
            end = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            b = data.alpaca_bars([sym], start=start, end=end, progress=False)
            self.daily[key] = pd.DataFrame({f: b[f][sym] for f in data.FIELDS}).dropna() \
                if sym in b["close"] else pd.DataFrame(columns=list(data.FIELDS))
        return self.daily[key]


# ---------- JSON payloads ----------

def scan_payload(store: Store, asof: pd.Timestamp, live: bool, p: ScanParams) -> dict:
    v = store.view(asof, p)
    alerted = {}
    state = store.cache_dir / "alerts.json"
    if state.exists():
        seen = json.loads(state.read_text())
        alerted = {k.split("/")[1]: v_ for k, v_ in seen.items() if k.startswith(f"{asof:%Y-%m-%d}/")}
    return {
        "asof": f"{asof:%Y-%m-%d %H:%M}", "live": live, "minutes": v["minutes"],
        "market": {k: ([{a: _num(b) for a, b in r.items()} for r in x] if k == "indexes" else _num(x))
                   for k, x in v["market"].items()},
        "universe": {k: getattr(p, k) for k in UNIVERSE_KEYS}, "scanned": len(v["rows"]),
        "rows": [{k: _num(x) for k, x in r.items()} for r in v["rows"]],
        "watchlist": [{k: _num(x) for k, x in r.items()} for r in v["follow"]],
        "climbers": [{k: _num(x) for k, x in r.items()} for r in v["climbers"]],
        "pm_volume": v["pm_volume"], "alerted": alerted, "defaults": asdict(WEB_PARAMS),
    }


def chart_payload(store: Store, asof: pd.Timestamp, p: ScanParams, symbols: list[str]) -> dict:
    v = store.view(asof, p)
    out = {}
    chart_start = pd.Timestamp.combine(asof.date(), CHART_START).tz_localize(ET)
    for sym in symbols[:40]:
        sb = v["bars"].get(sym, pd.DataFrame())
        pb = v["pmbars"].get(sym, pd.DataFrame())
        if not len(sb) and not len(pb):
            continue
        b = pd.concat([x for x in (pb, sb) if len(x)])
        # VWAP and the EMAs are the session's, as the scan sees them: none before 9:30
        vwap = (sb["vwap"] * sb["volume"]).cumsum() / sb["volume"].cumsum() if len(sb) else pd.Series(dtype=float)
        ema_f = scanner.ema(sb["close"], p.fast_ema) if len(sb) else pd.Series(dtype=float)
        ema_s = scanner.ema(sb["close"], p.slow_ema) if len(sb) else pd.Series(dtype=float)
        pad = [None] * len(pb)
        line = lambda s_: pad + s_.round(4).tolist()
        out[sym] = {"t": [f"{x.tz_convert(ET):%H:%M}" for x in b.index],
                    # 5-minute slot from 7:00, so gaps in thin premarket trading keep their place
                    "slot": [int((x - chart_start).total_seconds() // 300) for x in b.index],
                    "o": b["open"].round(4).tolist(), "h": b["high"].round(4).tolist(),
                    "l": b["low"].round(4).tolist(), "c": b["close"].round(4).tolist(),
                    "v": b["volume"].astype(float).tolist(), "vwap": line(vwap),
                    "ema_fast": line(ema_f), "ema_slow": line(ema_s)}
        out[sym]["day_volume"] = v["day_volume"].get(sym)
        if sym in v["stats"].index:
            out[sym]["prev_close"] = float(v["stats"].loc[sym, "prev_close"])
            out[sym]["avg_volume"] = float(v["stats"].loc[sym, "avg_volume"])
        elif sym in INDEX_FUNDS:
            out[sym]["prev_close"] = store.prev_close(sym, asof.normalize())
        if sym in v["watchlist"]:
            out[sym]["pm_high"] = v["watchlist"][sym]["pm_high"]
            out[sym]["pm_low"] = v["watchlist"][sym]["pm_low"]
            out[sym]["prev_close"] = v["watchlist"][sym]["prev_close"]
    return out


def daily_payload(store: Store, asof: pd.Timestamp, sym: str, p: ScanParams = WEB_PARAMS) -> dict:
    d = store.daily_bars(sym, asof.normalize())
    b = store.view(asof, p)["bars"].get(sym)
    if b is not None and len(b):          # today so far, from the 5-minute bars
        today = pd.DataFrame({"open": b["open"].iloc[0], "high": b["high"].max(), "low": b["low"].min(),
                              "close": b["close"].iloc[-1], "volume": b["volume"].sum()},
                             index=[asof.normalize().tz_localize(None)])
        d = pd.concat([d[d.index < today.index[0]], today])
    d = d.iloc[-90:]
    c = d["close"]
    return {"t": [f"{x:%m-%d}" for x in d.index], "o": d["open"].round(4).tolist(),
            "h": d["high"].round(4).tolist(), "l": d["low"].round(4).tolist(),
            "c": c.round(4).tolist(), "v": d["volume"].astype(float).tolist(),
            "sma20": c.rolling(20).mean().round(4).where(lambda s: s.notna(), None).tolist(),
            "sma50": c.rolling(50).mean().round(4).where(lambda s: s.notna(), None).tolist()}


def params_from(q: dict) -> ScanParams:
    p = WEB_PARAMS
    kw = {}
    for k in UNIVERSE_KEYS:
        if k in q:
            kw[k] = float(q[k][0])
    return replace(p, **kw)


def resolve_asof(q: dict) -> tuple[pd.Timestamp, bool]:
    at = (q.get("at") or [""])[0].strip()
    if not at:
        return latest_asof()
    asof = _floor5(pd.Timestamp(at, tz=ET))
    start = pd.Timestamp.combine(asof.date(), CHART_START).tz_localize(ET)
    close = pd.Timestamp.combine(asof.date(), CLOSE).tz_localize(ET)
    if not start + BAR <= asof <= close:
        raise ValueError(f"{asof:%Y-%m-%d %H:%M} ET is outside the hours shown (7:05 am-4:00 pm).")
    return asof, False


def _clean(o):
    """NaN -> null so the page gets valid JSON."""
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(x) for k, x in o.items()}
    if isinstance(o, list):
        return [_clean(x) for x in o]
    return o


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, HEAD.encode() + PAGE.read_bytes(), "text/html; charset=utf-8")
                asof, live = resolve_asof(q)
                p = params_from(q)
                if u.path == "/api/scan":
                    out = scan_payload(store, asof, live, p)
                elif u.path == "/api/charts":
                    syms = [s for s in (q.get("symbols") or [""])[0].upper().split(",") if s]
                    out = chart_payload(store, asof, p, syms)
                elif u.path == "/api/daily":
                    out = daily_payload(store, asof, q["symbol"][0].upper(), p)
                else:
                    return self._send(404, b'{"error": "not found"}')
                self._send(200, json.dumps(_clean(out)).encode())
            except (ValueError, SystemExit) as e:
                self._send(400, json.dumps({"error": str(e)}).encode())
            except Exception as e:            # keep serving; show the problem in the page
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())

        def log_message(self, fmt, *args):
            pass

    return Handler


def snapshot(store: Store, at: str | None, path: Path, max_charts: int = 60):
    """Save one view (scan + charts for the stocks nearest to matching) for
    viewing without the server: JSON, or with a .html name, the page itself."""
    q = {"at": [at]} if at else {}
    asof, live = resolve_asof(q)
    p = WEB_PARAMS
    scan = scan_payload(store, asof, live, p)

    def near(r):          # how many checks fail with default settings
        price, atr = r["price"], r["atr"]
        fails = [not (p.min_price <= price < p.max_price), r["rel_volume"] < p.min_rel_volume,
                 not (p.min_change <= r["change"] <= p.max_change), price <= r["vwap"],
                 not r["vwap_rising"], not r["ema_trend"],
                 price - r["vwap"] > p.max_atr_above_vwap * atr,
                 not (p.min_atr_pct <= atr / price <= p.max_atr_pct)]
        return sum(fails), -r["rel_volume"]
    picks = [r["symbol"] for r in sorted(scan["rows"], key=near)[:max_charts]]
    picks = [c["symbol"] for c in scan["climbers"]] + picks
    picks += [w["symbol"] for w in scan["watchlist"]] + list(INDEX_FUNDS)
    charts = {}
    for i in range(0, len(picks), 40):
        charts.update(chart_payload(store, asof, p, picks[i:i + 40]))
    daily = {s: daily_payload(store, asof, s) for s in picks[:20]}
    snap = json.dumps(_clean({"scan": scan, "charts": charts, "daily": daily}))
    if path.suffix == ".html":            # a page that opens without the server
        page = PAGE.read_text()
        i = page.index("<script>")
        snap = HEAD + page[:i] + "<script>window.SNAPSHOT = " + snap.replace("</", "<\\/") + ";</script>\n" + page[i:]
    path.write_text(snap)
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB, {len(charts)} charts) as of {scan['asof']} ET")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trendbt.webapp")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cache", default="data/scan",
                    help="the scanner's cache folder (daily stats, watchlists, alerts.json)")
    ap.add_argument("--snapshot", metavar="FILE", help="save one view (.html page or .json) and exit")
    ap.add_argument("--at", help='with --snapshot: a past moment, ET, e.g. "2026-10-14 14:00"')
    args = ap.parse_args(argv)
    data._alpaca_headers()                # fail early without keys
    store = Store(Path(args.cache))
    if args.snapshot:
        return snapshot(store, args.at, Path(args.snapshot))
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Trend dashboard on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
