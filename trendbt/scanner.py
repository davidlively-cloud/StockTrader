"""Intraday scanner: stocks in a healthy intraday uptrend, for day trading.

  python -m trendbt.scanner                         # scan now (market hours)
  python -m trendbt.scanner --at "2026-10-12 11:00" # replay a past moment (ET)
  python -m trendbt.scanner --state alerts.json     # only print symbols not alerted today

Data: Alpaca's free plan gives the consolidated (SIP) tape only 15 minutes late;
real-time data is IEX alone, about 2-3% of US volume. So every check runs on
SIP 5-minute bars that end 15 minutes ago, and the current price is the IEX
last trade, shown for reference only.

A stock is a match when all checks pass (defaults in ScanParams):
  liquid      price >= $5, 20-day average dollar volume >= $20M
  in play     today's volume >= 1.5x normal for this time of day
  up          +2% to +20% from yesterday's close
  trending    above a rising VWAP, 9-bar EMA above 20-bar EMA on 5-minute bars
  not chased  no more than half a day's average range (ATR) above VWAP
  tradable    daily ATR between 2% and 12% of price; spread <= 0.25% (live only)

Each match gets a plan: stop just under the higher of VWAP and the last
30 minutes' low (but at least 0.25 ATR below price), target at 2R, and a share count risking 1% of the account.
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
    min_rel_volume: float = 1.5
    min_change: float = 0.02             # from yesterday's close
    max_change: float = 0.20
    fast_ema: int = 9                    # on 5-minute bars
    slow_ema: int = 20
    vwap_rising_bars: int = 6            # VWAP higher than 30 minutes ago
    max_atr_above_vwap: float = 0.5
    min_atr_pct: float = 0.02
    max_atr_pct: float = 0.12
    max_spread_pct: float = 0.0025
    stop_lookback_bars: int = 6
    min_stop_atr: float = 0.25           # stop at least this many daily ATRs below price
    target_r: float = 2.0
    equity: float = 25_000.0
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.25
    top: int = 10


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
    return hits.head(p.top), market


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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trendbt.scanner")
    ap.add_argument("--at", help='replay a past moment, ET, e.g. "2026-10-12 11:00"')
    ap.add_argument("--cache", default="data/scan", help="folder for per-day daily stats")
    ap.add_argument("--state", help="JSON file of symbols already alerted; print only new ones")
    ap.add_argument("--json", action="store_true", help="print matches as JSON")
    ap.add_argument("--equity", type=float, default=25_000)
    args = ap.parse_args(argv)
    at = pd.Timestamp(args.at, tz=ET) if args.at else None
    hits, market = scan(at, ScanParams(equity=args.equity), Path(args.cache))

    if args.state:
        path = Path(args.state)
        day = f"{market['asof']:%Y-%m-%d}"
        seen = json.loads(path.read_text()) if path.exists() else {}
        done = set(seen.get(day, []))
        hits = hits[~hits["symbol"].isin(done)]
        path.write_text(json.dumps({day: sorted(done | set(hits["symbol"]))}))
        market["new"] = len(hits)

    if args.json:
        print(json.dumps({"market": market, "matches": hits.to_dict("records")},
                         default=str, indent=1))
    else:
        print(report(hits, market))


if __name__ == "__main__":
    main()
