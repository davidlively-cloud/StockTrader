# trendbt: US stock trend-following backtester

Implements the Stock Trend Prototype Plan: daily bars, trend filter plus one of
three entry signals, ATR-based stops, and a walk-forward test. Signals are taken
at the close and filled at the next open; stops that gap are filled at the open.

## Setup

```
pip install -r requirements.txt
export ALPACA_API_KEY_ID=...        # free Alpaca account, no funding needed
export ALPACA_API_SECRET_KEY=...
```

## Use

```
python -m trendbt fetch                     # every US stock since 2016 into data/
python -m trendbt fetch --symbols SPY,AAPL,MSFT,NVDA   # or a small list
python -m trendbt compare                   # 4 entry types x 3 exit sets, 2016-2023
python -m trendbt backtest --entry breakout # in-sample and held-out 2024+ results
python -m trendbt walkforward --entry any   # tune 3 years, test 6 months, roll
python -m trendbt demo                      # synthetic data; checks the engine only
python -m pytest tests                      # engine tests
```

### Intraday scanner

```
python -m trendbt.scanner                          # scan now, during market hours
python -m trendbt.scanner --at "2026-10-12 11:00"  # replay a past moment (ET)
python -m trendbt.scanner --state alerts.json      # list only symbols not alerted today
python -m trendbt.scanner --premarket              # gap-ups before the open (4:00-9:30 ET)
```

Finds liquid stocks in a healthy intraday uptrend while SPY is above its VWAP
(2x normal volume, up 4-20% on the day, above a rising VWAP, 9 EMA over 20 EMA on 5-minute bars, not
stretched far above VWAP) and prints a stop, a 2R target and a share count for
each. The criteria are listed at the top of `trendbt/scanner.py`. On Alpaca's
free plan the full market tape is 15 minutes late, so live scans judge the
trend as of 15 minutes ago and show the real-time IEX price beside it.

### Trend Desk (web dashboard)

```
python -m trendbt.webapp                   # then open http://127.0.0.1:8050
python -m trendbt.webapp --cache data/scan # same cache folder as the scanner
python -m trendbt.webapp --snapshot view.html   # save the current view as a page
```

One page with what the alert thread posts: SPY against its VWAP (plus the Dow
and Nasdaq Composite through their tracking funds DIA and ONEQ), the trend
alerts with stop, target and share count, the premarket watchlist with each
name's status (breaking out, holding, fading, failed), names that miss by one
check, and a 5-minute chart (VWAP, 9/20 EMA, stop, target, premarket high and
low) for every stock listed, plus a 90-day chart on click. Each 5-minute chart
starts at 7:00 am ET with the premarket trading shaded; VWAP and the EMAs start
at the 9:30 open. Before the open the page shows today's premarket so far. The scan settings
sit in the left rail: trend and trade-plan settings re-filter instantly in the
browser; the price and dollar-volume settings re-download (about 30 seconds).
"Look back to" replays any 5-minute moment of a past session. The live view
refreshes every 5 minutes. No extra packages: the server is the Python
standard library.

To see it on a phone, `.github/workflows/trend-desk.yml` publishes a saved
view to GitHub Pages every 15 minutes on weekdays from 7:00 am ET. It needs the Alpaca keys as
repository secrets (`ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`) and
Settings > Pages > Source set to "GitHub Actions". The keys stay in GitHub;
the page holds only market data. Anyone with the link can open the page while
the repository is public.

Results land in `results/` (equity curve, trade list, summary per run).

## Layout

| File | What it does |
| --- | --- |
| `trendbt/config.py` | Every setting, with the plan's defaults |
| `trendbt/data.py` | Alpaca download, yfinance/CSV fallbacks, Parquet cache |
| `trendbt/indicators.py` | SMA, ATR, ADX, RSI (Wilder smoothing) |
| `trendbt/signals.py` | Universe, market regime, trend filter, entry signals |
| `trendbt/simulator.py` | Day-by-day portfolio with sizing, stops, targets, time stop |
| `trendbt/metrics.py` | CAGR, drawdown, Sharpe, win rate, profit factor, R-multiples |
| `trendbt/walkforward.py` | Grid search and walk-forward |
| `trendbt/scanner.py` | Intraday scanner for day-trade setups |
| `trendbt/webapp.py`, `trendbt/web/index.html` | Trend Desk web dashboard for the scanner |

## Known limits

- Alpaca's coverage of delisted stocks is partial, so results can be flattered
  by survivorship bias. Check any promising result against a bias-free source.
- Stops are checked once per day against the low; same-day target and stop
  hits assume the stop came first.
