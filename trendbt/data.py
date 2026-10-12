"""Market data loading and local caching.

Sources:
  alpaca   Free Alpaca account; needs ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.
           Split- and dividend-adjusted daily bars from all US exchanges since 2016.
  yfinance Unofficial Yahoo Finance data; for quick checks on a ticker list only.
  csv      A folder of SYMBOL.csv files with date,open,high,low,close,volume.

Bars are stored as wide Parquet files (rows = dates, columns = symbols), one
per field, under the cache directory.
"""
import os
import time
from pathlib import Path

import pandas as pd
import requests

FIELDS = ("open", "high", "low", "close", "volume")
ALPACA_DATA = "https://data.alpaca.markets/v2"
ALPACA_TRADING = "https://paper-api.alpaca.markets/v2"


# ---------- cache ----------

try:
    import pyarrow  # noqa: F401
    _EXT = "parquet"
except ImportError:          # some environments can't install pyarrow; pickle works anywhere
    _EXT = "pkl"


def save(bars: dict[str, pd.DataFrame], cache_dir: str | Path):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for f in FIELDS:
        if _EXT == "parquet":
            bars[f].to_parquet(cache_dir / f"{f}.parquet")
        else:
            bars[f].to_pickle(cache_dir / f"{f}.pkl")


def load(cache_dir: str | Path) -> dict[str, pd.DataFrame]:
    cache_dir = Path(cache_dir)
    for ext, reader in (("parquet", pd.read_parquet), ("pkl", pd.read_pickle)):
        if (cache_dir / f"close.{ext}").exists():
            return {f: reader(cache_dir / f"{f}.{ext}") for f in FIELDS}
    raise SystemExit(f"No data in {cache_dir}/. Run `python -m trendbt fetch` first.")


def to_wide(long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """long: columns symbol, date, open, high, low, close, volume."""
    long = long.drop_duplicates(["date", "symbol"]).sort_values("date")
    return {f: long.pivot(index="date", columns="symbol", values=f).sort_index()
            for f in FIELDS}


# ---------- Alpaca ----------

def _alpaca_headers():
    try:
        return {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
                "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"]}
    except KeyError as e:
        raise SystemExit(f"Set {e.args[0]} (from your free Alpaca account) first.") from None


def _get(url, params, headers, retries=5):
    for attempt in range(retries):
        r = requests.get(url, params=params, headers=headers, timeout=60)
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def alpaca_symbols(include_inactive=True) -> list[str]:
    """US equities on major exchanges. Inactive (delisted) assets are included
    to reduce survivorship bias, though Alpaca's delisted coverage is partial."""
    h = _alpaca_headers()
    syms = []
    for status in ("active", "inactive") if include_inactive else ("active",):
        assets = _get(f"{ALPACA_TRADING}/assets",
                      {"status": status, "asset_class": "us_equity"}, h)
        syms += [a["symbol"] for a in assets
                 if a.get("exchange") in ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS")
                 and "." not in a["symbol"] and "/" not in a["symbol"]]
    return sorted(set(syms))


def alpaca_bars(symbols: list[str], start="2016-01-01", end=None,
                batch=100, progress=True) -> dict[str, pd.DataFrame]:
    h = _alpaca_headers()
    end = end or (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rows = []
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        params = {"symbols": ",".join(chunk), "timeframe": "1Day", "start": start,
                  "end": end, "adjustment": "all", "feed": "sip", "limit": 10000}
        while True:
            js = _get(f"{ALPACA_DATA}/stocks/bars", params, h)
            for sym, bars in (js.get("bars") or {}).items():
                for b in bars:
                    rows.append((sym, b["t"][:10], b["o"], b["h"], b["l"], b["c"], b["v"]))
            tok = js.get("next_page_token")
            if not tok:
                break
            params["page_token"] = tok
        if progress:
            print(f"  {min(i + batch, len(symbols))}/{len(symbols)} symbols, {len(rows):,} bars")
    long = pd.DataFrame(rows, columns=["symbol", "date", *FIELDS])
    long["date"] = pd.to_datetime(long["date"])
    return to_wide(long)


# ---------- other sources ----------

def yfinance_bars(symbols: list[str], start="2016-01-01", end=None):
    import yfinance as yf  # optional dependency
    df = yf.download(symbols, start=start, end=end, auto_adjust=True,
                     group_by="column", progress=False)
    return {f: df[f.capitalize()].copy() for f in FIELDS}


def csv_bars(folder: str | Path):
    frames = []
    for path in Path(folder).glob("*.csv"):
        d = pd.read_csv(path, parse_dates=["date"])
        d.columns = [c.lower() for c in d.columns]
        d["symbol"] = path.stem.upper()
        frames.append(d[["symbol", "date", *FIELDS]])
    return to_wide(pd.concat(frames))
