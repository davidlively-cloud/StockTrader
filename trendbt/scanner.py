"""Intraday scanner: stocks in a healthy intraday uptrend, for day trading.

  python -m trendbt.scanner                         # scan now (market hours)
  python -m trendbt.scanner --at "2026-10-12 11:00" # replay a past moment (ET)
  python -m trendbt.scanner --state alerts.json     # only print symbols not alerted today
  python -m trendbt.scanner --replay 2026-10-12     # a day's alerts and how each played out
  python -m trendbt.scanner --premarket             # gap-ups before the open (4:00-9:30 ET)
  python -m trendbt.scanner --follow                # scan, plus how the premarket names are doing

Data: Alpaca's free plan gives the consolidated (SIP) tape only 15 minutes late;
real-time data is IEX alone, about 2-3% of US volume. So every check runs on
SIP 5-minute bars that end 15 minutes ago, and the current price is the IEX
last trade, shown for reference only.

A stock is a match when all checks pass (defaults in ScanParams):
  liquid      price >= $5, 20-day average dollar volume >= $20M
  market      SPY above its VWAP
  in play     today's volume >= 2x normal for this time of day
  up          +4% to +20% from yesterday's close
  trending    above a rising VWAP, 9-bar EMA above 20-bar EMA on 5-minute bars
  not chased  no more than half a day's average range (ATR) above VWAP
  tradable    daily ATR between 2% and 12% of price; spread <= 0.25% (live only)

Each match gets a plan: stop just under the higher of VWAP and the last
30 minutes' low (but at least 0.75 ATR below price), target at 2R, and a share count risking 1% of the account.
"""
import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import data

ET = ZoneInfo("America/New_York")
OPEN, CLOSE = dtime(9, 30), dtime(16, 0)
SIP_DELAY = pd.Timedelta(minutes=15)

# Typical share of a day's volume traded by N minutes after the open (US stocks,
# U-shaped intraday profile). Used to judge relative volume mid-session.
VOLUME_CURVE = [(0, 0.0), (15, 0.09), (30, 0.15), (60, 0.25), (120, 0.40),
                (180, 0.52), (240, 0.63), (300, 0.74), (360, 0.87), (390, 1.0)]

# Funds and other non-operating-company listings, by name
FUND_RE = re.compile(r"\b(ETF|ETN|Fund|Trust|ProShares|Direxion|iShares|"
                     r"2X|3X|Leveraged|Inverse|Bull|Bear|Warrants?|Units?|Rights?|"
                     r"Acquisition)\b", re.I)


@dataclass(frozen=True)
class ScanParams:
    min_price: float = 5.0
    min_dollar_volume: float = 20e6      # 20-day average
    min_rel_volume: float = 2.0
    min_change: float = 0.04             # from yesterday's close
    max_change: float = 0.20
    fast_ema: int = 9                    # on 5-minute bars
    slow_ema: int = 20
    vwap_rising_bars: int = 6            # VWAP higher than 30 minutes ago
    max_atr_above_vwap: float = 0.5
    min_atr_pct: float = 0.02
    max_atr_pct: float = 0.12
    max_spread_pct: float = 0.0025
    stop_lookback_bars: int = 6
    min_stop_atr: float = 0.75           # stop at least this many daily ATRs below price
    target_r: float = 2.0
    equity: float = 25_000.0
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.25
    top: int = 10
    require_market_up: bool = True       # only alert while SPY is above its VWAP


def volume_fraction(minutes: float) -> float:
    xs, ys = zip(*VOLUME_CURVE)
    return float(np.interp(minutes, xs, ys))


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def evaluate(bars: pd.DataFrame, prev_close: float, avg_volume: float,
             daily_atr: float, minutes: float, p: ScanParams) -> dict | None:
    """Score one stock. bars: today's 5-minute bars (open, high, low, close,
    volume, vwap) up to the scan time. Returns the checks and a trade plan,
    or None when there are too few bars to judge."""
    if len(bars) < max(p.fast_ema, p.vwap_rising_bars + 1) or not (prev_close > 0 and avg_volume > 0 and daily_atr > 0):
        return None
    close = bars["close"]
    price = float(close.iloc[-1])
    cum_vol = bars["volume"].cumsum()
    vwap = (bars["vwap"] * bars["volume"]).cumsum() / cum_vol
    v = float(vwap.iloc[-1])
    fast, slow = ema(close, p.fast_ema), ema(close, p.slow_ema)
    change = price / prev_close - 1
    rel_vol = float(cum_vol.iloc[-1]) / (avg_volume * max(volume_fraction(minutes), 0.05))
    atr_pct = daily_atr / price
    checks = {
        "liquid": price >= p.min_price and avg_volume * price >= p.min_dollar_volume,
        "in_play": rel_vol >= p.min_rel_volume,
        "up": p.min_change <= change <= p.max_change,
        "above_vwap": price > v,
        "vwap_rising": v > float(vwap.iloc[-1 - p.vwap_rising_bars]),
        "ema_trend": fast.iloc[-1] > slow.iloc[-1] and price >= fast.iloc[-1],
        "not_chased": price - v <= p.max_atr_above_vwap * daily_atr,
        "atr_ok": p.min_atr_pct <= atr_pct <= p.max_atr_pct,
    }
    stop = max(v, float(bars["low"].iloc[-p.stop_lookback_bars:].min())) - 0.05 * daily_atr
    stop = min(stop, price - p.min_stop_atr * daily_atr)   # room for 5-minute noise
    risk = price - stop
    shares = 0
    if risk > 0:
        shares = int(min(p.equity * p.risk_per_trade / risk,
                         p.equity * p.max_position_pct / price))
    return {
        "price": price, "change": change, "rel_volume": rel_vol, "vwap": v,
        "atr_pct": atr_pct, "stop": stop, "target": price + p.target_r * risk,
        "risk_pct": risk / price, "shares": shares, "checks": checks,
        "match": all(checks.values()) and risk > 0,
        # rank: how unusual the volume is, times how cleanly price holds over VWAP
        "score": rel_vol * min((price - v) / daily_atr + 0.1, 1.0),
    }


# ---------- data ----------

def tradable_stocks() -> list[str]:
    h = data._alpaca_headers()
    assets = data._get(f"{data.ALPACA_TRADING}/assets",
                       {"status": "active", "asset_class": "us_equity"}, h)
    return sorted(a["symbol"] for a in assets
                  if a.get("tradable") and a.get("exchange") in ("NYSE", "NASDAQ", "AMEX")
                  and re.fullmatch(r"[A-Z]{1,5}", a["symbol"])
                  and not FUND_RE.search(a.get("name") or ""))


def daily_stats(symbols, day: pd.Timestamp, cache_dir: Path | None) -> pd.DataFrame:
    """Per symbol: yesterday's close, 20-day average volume, 14-day ATR, all
    from sessions before `day`. Cached per day since it does not change."""
    path = cache_dir / f"daily_{day:%Y-%m-%d}.pkl" if cache_dir else None
    if path and path.exists():
        return pd.read_pickle(path)
    start = (day - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    end = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    b = data.alpaca_bars(symbols, start=start, end=end, batch=200, progress=False)
    b = {f: x[x.index < day.tz_localize(None)] for f, x in b.items()}
    prev = b["close"].shift(1)
    tr = np.fmax(np.fmax(b["high"] - b["low"], (b["high"] - prev).abs()), (b["low"] - prev).abs())
    out = pd.DataFrame({
        "prev_close": b["close"].ffill().iloc[-1],
        "avg_volume": b["volume"].iloc[-20:].mean(),
        "atr": tr.iloc[-14:].mean(),
        "days": b["close"].iloc[-20:].count(),
    }).dropna()
    out = out[out["days"] >= 15]
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_pickle(path)
    return out


def intraday_bars(symbols, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    h = data._alpaca_headers()
    rows = []
    for i in range(0, len(symbols), 200):
        params = {"symbols": ",".join(symbols[i:i + 200]), "timeframe": "5Min",
                  "start": start.tz_convert("UTC").isoformat(),
                  "end": end.tz_convert("UTC").isoformat(),
                  "adjustment": "raw", "feed": "sip", "limit": 10000}
        while True:
            js = data._get(f"{data.ALPACA_DATA}/stocks/bars", params, h)
            for sym, bars in (js.get("bars") or {}).items():
                rows += [(sym, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"], b.get("vw", b["c"]))
                         for b in bars]
            if not js.get("next_page_token"):
                break
            params["page_token"] = js["next_page_token"]
    df = pd.DataFrame(rows, columns=["symbol", "t", "open", "high", "low", "close", "volume", "vwap"])
    df["t"] = pd.to_datetime(df["t"])
    return {s: g.set_index("t").drop(columns="symbol") for s, g in df.groupby("symbol")}


def live_quotes(symbols) -> dict[str, dict]:
    """Delayed NBBO spread (SIP) and real-time last trade (IEX)."""
    h = data._alpaca_headers()
    out = {s: {} for s in symbols}
    for feed in ("delayed_sip", "iex"):
        for i in range(0, len(symbols), 1000):
            js = data._get(f"{data.ALPACA_DATA}/stocks/snapshots",
                           {"symbols": ",".join(symbols[i:i + 1000]), "feed": feed}, h)
            for s, snap in js.items():
                if feed == "delayed_sip":
                    q = snap.get("latestQuote") or {}
                    if q.get("ap") and q.get("bp"):
                        out[s]["spread"] = (q["ap"] - q["bp"]) / ((q["ap"] + q["bp"]) / 2)
                else:
                    out[s]["iex_last"] = (snap.get("latestTrade") or {}).get("p")
    return out


# ---------- scan ----------

def scan(at: pd.Timestamp | None = None, p: ScanParams = ScanParams(),
         cache_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    live = at is None
    now = pd.Timestamp.now(tz=ET) if live else at
    asof = now - SIP_DELAY if live else now
    day = asof.normalize()
    session_open = pd.Timestamp.combine(day.date(), OPEN).tz_localize(ET)
    minutes = (asof - session_open).total_seconds() / 60
    if not 0 < minutes <= 390:
        raise SystemExit(f"{asof:%Y-%m-%d %H:%M} ET is outside regular trading hours.")

    stats = daily_stats(tradable_stocks(), day, cache_dir)
    stats = stats[(stats["prev_close"] >= p.min_price)
                  & (stats["prev_close"] * stats["avg_volume"] >= p.min_dollar_volume)]
    bars = intraday_bars(sorted(set(stats.index) | {"SPY"}), session_open, asof)
    # a 5-minute bar is stamped with its start; keep only bars finished by `asof`
    bars = {s: b[b.index + pd.Timedelta(minutes=5) <= asof] for s, b in bars.items()}

    rows = []
    for sym, st in stats.iterrows():
        r = evaluate(bars.get(sym, pd.DataFrame()), st.prev_close, st.avg_volume,
                     st.atr, minutes, p)
        if r:
            rows.append({"symbol": sym, **{k: v for k, v in r.items() if k != "checks"},
                         **r["checks"]})
    res = pd.DataFrame(rows)
    hits = (res[res["match"]].sort_values("score", ascending=False)
            if len(res) else res)
    if live and len(hits):
        q = live_quotes(list(hits["symbol"]))
        hits = hits.assign(spread=[q[s].get("spread", np.nan) for s in hits["symbol"]],
                           iex_last=[q[s].get("iex_last") for s in hits["symbol"]])
        hits = hits[~(hits["spread"] > p.max_spread_pct)]

    spy = bars.get("SPY")
    market = {"asof": asof, "scanned": len(res), "matches": len(hits)}
    if spy is not None and len(spy):
        vw = float((spy["vwap"] * spy["volume"]).sum() / spy["volume"].sum())
        market.update(spy=float(spy["close"].iloc[-1]), spy_vwap=vw,
                      spy_above_vwap=float(spy["close"].iloc[-1]) > vw)
        if p.require_market_up and not market["spy_above_vwap"]:
            hits = hits.iloc[0:0]
    return hits.head(p.top), market


# ---------- premarket ----------

PREMARKET_OPEN = dtime(4, 0)


@dataclass(frozen=True)
class PremarketParams:
    min_price: float = 5.0
    min_dollar_volume: float = 20e6      # 20-day average, regular session
    min_gap: float = 0.03                # premarket last vs yesterday's close
    max_gap: float = 0.30
    min_pm_dollar_volume: float = 1e6    # traded so far this morning
    min_pm_volume_pct: float = 0.02      # of a normal day's volume
    max_fade: float = 0.35               # gave back at most this share of the gap from the high
    top: int = 10


def premarket(at: pd.Timestamp | None = None, p: PremarketParams = PremarketParams(),
              cache_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Stocks gapping up before the open on unusual volume and holding the gap:
    a watchlist for the open, with the premarket high and low as key levels."""
    now = pd.Timestamp.now(tz=ET) if at is None else at
    asof = now - SIP_DELAY if at is None else now
    day = asof.normalize()
    start = pd.Timestamp.combine(day.date(), PREMARKET_OPEN).tz_localize(ET)
    if not start < asof <= pd.Timestamp.combine(day.date(), OPEN).tz_localize(ET):
        raise SystemExit(f"{asof:%Y-%m-%d %H:%M} ET is outside premarket hours (4:00-9:30).")
    stats = daily_stats(tradable_stocks(), day, cache_dir)
    stats = stats[(stats["prev_close"] >= p.min_price)
                  & (stats["prev_close"] * stats["avg_volume"] >= p.min_dollar_volume)]
    bars = intraday_bars(list(stats.index), start, asof)
    rows = []
    for sym, b in bars.items():
        b = b[b.index + pd.Timedelta(minutes=5) <= asof]
        if not len(b):
            continue
        st = stats.loc[sym]
        last, hi, lo = float(b["close"].iloc[-1]), float(b["high"].max()), float(b["low"].min())
        vol = float(b["volume"].sum())
        gap = last / st.prev_close - 1
        fade = (hi - last) / (hi - st.prev_close) if hi > st.prev_close else 1.0
        if (p.min_gap <= gap <= p.max_gap and vol * last >= p.min_pm_dollar_volume
                and vol >= p.min_pm_volume_pct * st.avg_volume and fade <= p.max_fade):
            rows.append({"symbol": sym, "last": last, "gap": gap, "pm_high": hi, "pm_low": lo,
                         "prev_close": st.prev_close, "pm_volume": vol,
                         "pm_volume_pct": vol / st.avg_volume, "atr_pct": st.atr / last})
    res = pd.DataFrame(rows)
    if len(res):
        res = res.sort_values("pm_volume_pct", ascending=False)
    return res.head(p.top), {"asof": asof, "scanned": len(stats), "matches": len(res)}


def premarket_report(hits: pd.DataFrame, info: dict) -> str:
    lines = [f"Premarket gappers as of {info['asof']:%a %b %d %H:%M} ET (SIP, 15 min delayed live): "
             f"{info['matches']} of {info['scanned']} liquid stocks."]
    for _, r in hits.iterrows():
        lines.append(f"{r['symbol']:<5} {r['last']:>8.2f} (gap {r['gap']:+.1%}) "
                     f"premarket vol {r['pm_volume_pct']:.0%} of a normal day | "
                     f"PM high {r['pm_high']:.2f}, PM low {r['pm_low']:.2f}")
    if len(hits):
        lines.append("Watchlist only: a common entry is a break above the PM high after the open, "
                     "with a stop under the PM low or VWAP. Regular scans confirm trends from 10:30.")
    return "\n".join(lines)


def save_watchlist(hits: pd.DataFrame, info: dict, cache_dir: Path):
    """Remember today's premarket names and levels so the day's scans can follow them."""
    path = cache_dir / f"watchlist_{info['asof']:%Y-%m-%d}.json"
    wl = json.loads(path.read_text()) if path.exists() else {}
    for _, r in hits.iterrows():
        wl[r["symbol"]] = {k: float(r[k]) for k in ("prev_close", "pm_high", "pm_low", "gap")}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(wl, indent=1))


def follow(at: pd.Timestamp | None = None, cache_dir: Path = Path("data/scan")) -> tuple[pd.DataFrame, dict]:
    """How this morning's premarket names are trading now: price against the
    open, session VWAP and the premarket high and low."""
    now = pd.Timestamp.now(tz=ET) if at is None else at
    asof = now - SIP_DELAY if at is None else now
    path = cache_dir / f"watchlist_{asof:%Y-%m-%d}.json"
    session_open = pd.Timestamp.combine(asof.date(), OPEN).tz_localize(ET)
    info = {"asof": asof, "names": 0}
    if not path.exists() or asof <= session_open + pd.Timedelta(minutes=5):
        return pd.DataFrame(), info
    wl = json.loads(path.read_text())
    info["names"] = len(wl)
    bars = intraday_bars(sorted(wl), session_open, asof)
    rows = []
    for sym, lv in wl.items():
        b = bars.get(sym)
        b = b[b.index + pd.Timedelta(minutes=5) <= asof] if b is not None else None
        if b is None or not len(b):
            continue
        last = float(b["close"].iloc[-1])
        vwap = float((b["vwap"] * b["volume"]).sum() / b["volume"].sum())
        if last < lv["pm_low"]:
            status = "failed (below PM low)"
        elif last > lv["pm_high"] and last > vwap:
            status = "breaking out (above PM high and VWAP)"
        elif last > vwap:
            status = "holding (above VWAP)"
        else:
            status = "fading (below VWAP)"
        rows.append({"symbol": sym, "last": last, "change": last / lv["prev_close"] - 1,
                     "from_open": last / float(b["open"].iloc[0]) - 1, "vwap": vwap,
                     "pm_high": lv["pm_high"], "day_high": float(b["high"].max()),
                     "status": status})
    df = pd.DataFrame(rows)
    return (df.sort_values("from_open", ascending=False) if len(df) else df), info


def follow_report(df: pd.DataFrame, info: dict) -> str:
    if not len(df):
        return ""
    lines = [f"Premarket names as of {info['asof']:%H:%M} ET:"]
    for _, r in df.iterrows():
        lines.append(f"{r['symbol']:<5} {r['last']:>8.2f} ({r['change']:+.1%} on day, "
                     f"{r['from_open']:+.1%} from open) VWAP {r['vwap']:.2f}, "
                     f"PM high {r['pm_high']:.2f} | {r['status']}")
    return "\n".join(lines)


def outcome(after: pd.DataFrame, entry: float, stop: float, target: float) -> tuple[str, float]:
    """First of stop or target hit in the bars after the alert, else exit at the
    last close. A bar touching both counts as the stop. Returns (exit, R)."""
    risk = entry - stop
    for _, b in after.iterrows():
        if b["low"] <= stop:
            return "stop", -1.0
        if b["high"] >= target:
            return "target", (target - entry) / risk
    return "close", (float(after["close"].iloc[-1]) - entry) / risk if len(after) else 0.0


def replay(day: str, times=("10:15", "10:45", "11:15", "11:45", "12:15", "12:45",
                            "13:15", "13:45", "14:15", "14:45", "15:15"),
           p: ScanParams = ScanParams(), cache_dir: Path | None = None,
           delay: pd.Timedelta = SIP_DELAY) -> pd.DataFrame:
    """Every alert a 30-minute scan would have sent on `day`, each symbol once,
    and how its plan played out by the close. Entry is the price `delay` after
    the bars the scan saw, which is what a live scan on delayed data gets."""
    d = pd.Timestamp(day, tz=ET)
    session_open = pd.Timestamp.combine(d.date(), OPEN).tz_localize(ET)
    stats = daily_stats(tradable_stocks(), d, cache_dir)
    stats = stats[(stats["prev_close"] >= p.min_price)
                  & (stats["prev_close"] * stats["avg_volume"] >= p.min_dollar_volume)]
    path = cache_dir / f"bars5_{day}.pkl" if cache_dir else None
    if path and path.exists():
        allbars = pd.read_pickle(path)
    else:
        allbars = intraday_bars(sorted(set(stats.index) | {"SPY"}), session_open,
                                pd.Timestamp.combine(d.date(), CLOSE).tz_localize(ET))
        if path:
            pd.to_pickle(allbars, path)
    spy = allbars.get("SPY")
    rows, seen = [], set()
    for t in times:
        asof = pd.Timestamp(f"{day} {t}", tz=ET)
        minutes = (asof - session_open).total_seconds() / 60
        market_up = market_ok(spy, asof)
        if p.require_market_up and not market_up:
            continue
        for sym, st in stats.iterrows():
            b = allbars.get(sym)
            if sym in seen or b is None:
                continue
            r = evaluate(b[b.index + pd.Timedelta(minutes=5) <= asof], st.prev_close,
                         st.avg_volume, st.atr, minutes, p)
            if not (r and r["match"]):
                continue
            seen.add(sym)
            later = b[b.index >= asof + delay]
            entry = float(later["open"].iloc[0]) if len(later) else r["price"]
            # keep the scan's stop and R; a late entry above the stop shifts R
            if entry <= r["stop"]:
                continue
            target = entry + p.target_r * (entry - r["stop"])
            how, R = outcome(later, entry, r["stop"], target)
            rows.append({"day": day, "time": t, "symbol": sym, "signal": r["price"],
                         "entry": entry, "stop": r["stop"], "target": target,
                         "rel_volume": r["rel_volume"], "change": r["change"],
                         "market_up": market_up, "exit": how, "R": R})
    return pd.DataFrame(rows)


def market_ok(spy: pd.DataFrame | None, asof: pd.Timestamp) -> bool:
    """SPY above its session VWAP as of `asof`."""
    if spy is None:
        return True
    s = spy[spy.index + pd.Timedelta(minutes=5) <= asof]
    if not len(s):
        return True
    return float(s["close"].iloc[-1]) > float((s["vwap"] * s["volume"]).sum() / s["volume"].sum())


def report(hits: pd.DataFrame, market: dict) -> str:
    lines = [f"Scan as of {market['asof']:%a %b %d %H:%M} ET (SIP data, 15 min delayed live): "
             f"{market['matches']} of {market['scanned']} liquid stocks match"
             + (f", {market['new']} new since the last scan." if "new" in market else ".")]
    if "spy" in market:
        tone = "above" if market["spy_above_vwap"] else "BELOW (weak tape, size down)"
        lines.append(f"SPY {market['spy']:.2f}, {tone} its VWAP {market['spy_vwap']:.2f}.")
    for _, r in hits.iterrows():
        last = f", IEX last {r['iex_last']:.2f}" if pd.notna(r.get("iex_last", np.nan)) else ""
        lines.append(
            f"{r['symbol']:<5} {r['price']:>8.2f} ({r['change']:+.1%}{last}) "
            f"relvol {r['rel_volume']:.1f}x, VWAP {r['vwap']:.2f} | "
            f"stop {r['stop']:.2f} ({r['risk_pct']:.1%}), target {r['target']:.2f}, "
            f"{r['shares']} sh")
    return "\n".join(lines)


def _only_new(hits: pd.DataFrame, info: dict, path: Path, kind: str) -> pd.DataFrame:
    """Drop symbols already alerted today by this kind of scan and record the rest."""
    key = f"{info['asof']:%Y-%m-%d}/{kind}"
    seen = json.loads(path.read_text()) if path.exists() else {}
    done = set(seen.get(key, []))
    hits = hits[~hits["symbol"].isin(done)] if len(hits) else hits
    seen = {k: v for k, v in seen.items() if k.startswith(f"{info['asof']:%Y-%m-%d}/")}
    seen[key] = sorted(done | set(hits["symbol"] if len(hits) else []))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seen))
    info["new"] = len(hits)
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trendbt.scanner")
    ap.add_argument("--at", help='replay a past moment, ET, e.g. "2026-10-12 11:00"')
    ap.add_argument("--cache", default="data/scan", help="folder for per-day daily stats")
    ap.add_argument("--premarket", action="store_true", help="premarket gap scan (4:00-9:30 ET)")
    ap.add_argument("--follow", action="store_true",
                    help="also report how this morning's premarket names are trading")
    ap.add_argument("--replay", metavar="DAY", help="replay a whole day's 30-minute scans")
    ap.add_argument("--state", help="JSON file of symbols already alerted; print only new ones")
    ap.add_argument("--json", action="store_true", help="print matches as JSON")
    ap.add_argument("--equity", type=float, default=25_000)
    args = ap.parse_args(argv)
    if args.replay:
        df = replay(args.replay, p=ScanParams(equity=args.equity), cache_dir=Path(args.cache))
        print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}") if len(df) else "No alerts.")
        if len(df):
            print(f"\n{len(df)} alerts, {df['R'].sum():+.1f}R total, {(df['R'] > 0).mean():.0%} winners")
        return
    at = pd.Timestamp(args.at, tz=ET) if args.at else None
    if args.premarket:
        hits, info = premarket(at, cache_dir=Path(args.cache))
        save_watchlist(hits, info, Path(args.cache))
        if args.state:
            hits = _only_new(hits, info, Path(args.state), "premarket")
        print(premarket_report(hits, info))
        return
    hits, market = scan(at, ScanParams(equity=args.equity), Path(args.cache))

    if args.state:
        hits = _only_new(hits, market, Path(args.state), "session")

    if args.json:
        print(json.dumps({"market": market, "matches": hits.to_dict("records")},
                         default=str, indent=1))
    else:
        print(report(hits, market))
    if args.follow:
        fl = follow_report(*follow(at, Path(args.cache)))
        if fl:
            print("\n" + fl)


if __name__ == "__main__":
    main()
