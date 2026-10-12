"""Command line entry point.

  python -m trendbt fetch      --cache data/            # download from Alpaca
  python -m trendbt backtest   --cache data/ --entry breakout
  python -m trendbt compare    --cache data/            # every entry type, same exits
  python -m trendbt walkforward --cache data/
  python -m trendbt demo                                 # synthetic data, engine check only
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import data, simulator
from .config import Params
from .metrics import summarize, buy_and_hold
from .signals import prepare
from .walkforward import grid, walk_forward

TRAIN_END = "2023-12-31"    # tuning period ends here; 2024+ is held out


def _load(args):
    if args.cmd == "demo":
        from .synthetic import random_market
        return random_market()
    bars = data.load(args.cache)
    if "SPY" not in bars["close"]:
        raise SystemExit("SPY must be in the dataset (used for the market regime filter).")
    return bars, bars["close"]["SPY"]


def _fmt(s: dict) -> str:
    pct = {"total_return", "cagr", "max_drawdown", "win_rate", "avg_win_pct",
           "avg_loss_pct", "avg_exposure"}
    lines = []
    for k, v in s.items():
        if k in pct:
            v = f"{v:.1%}"
        elif isinstance(v, float):
            v = f"{v:,.2f}"
        lines.append(f"  {k:<16} {v}")
    return "\n".join(lines)


def cmd_fetch(args):
    syms = args.symbols.split(",") if args.symbols else data.alpaca_symbols()
    if "SPY" not in syms:
        syms.append("SPY")
    print(f"Downloading {len(syms)} symbols from Alpaca...")
    bars = data.alpaca_bars(syms, start=args.start)
    data.save(bars, args.cache)
    print(f"Saved {bars['close'].shape[1]} symbols x {bars['close'].shape[0]} days to {args.cache}")


def run_one(bars, spy, p, start, end, out_dir=None, label=None):
    res = simulator.run(prepare(bars, spy, p), p, start, end)
    s = summarize(res.equity, res.trades, res.exposure)
    bench = summarize(buy_and_hold(spy.loc[res.equity.index], p.initial_equity))
    if out_dir:
        out = Path(out_dir) / (label or p.entry)
        out.mkdir(parents=True, exist_ok=True)
        res.equity.to_csv(out / "equity.csv")
        res.trades.to_csv(out / "trades.csv", index=False)
        (out / "summary.json").write_text(json.dumps(
            {"params": p.to_dict(), "strategy": s, "spy_buy_hold": bench}, indent=2, default=str))
    return s, bench


def cmd_backtest(args):
    bars, spy = _load(args)
    p = Params(entry=args.entry, initial_equity=args.equity)
    for name, (a, b) in {"in-sample": (args.start, TRAIN_END),
                         "held-out": ("2024-01-01", None)}.items():
        s, bench = run_one(bars, spy, p, a, b, args.out, f"{p.entry}_{name}")
        print(f"\n{p.entry} {name}\n{_fmt(s)}\n  SPY CAGR {bench.get('cagr', 0):.1%}, "
              f"max DD {bench.get('max_drawdown', 0):.1%}")


def cmd_compare(args):
    bars, spy = _load(args)
    rows = []
    for entry in ("breakout", "pullback", "any", "momentum"):
        for exits in ("trail_only", "default", "trend_break"):
            p = Params(entry=entry, initial_equity=args.equity)
            if exits == "trail_only":
                p = p.with_(partial_target_r=0, breakeven_r=0, time_stop_days=0)
            elif exits == "trend_break":
                p = p.with_(trend_exit_ma=50, time_stop_days=0)
            s, bench = run_one(bars, spy, p, args.start, TRAIN_END, args.out, f"{entry}_{exits}")
            rows.append({"entry": entry, "exits": exits, **{k: s.get(k) for k in (
                "cagr", "max_drawdown", "sharpe", "trades_per_year", "win_rate",
                "win_loss_ratio", "avg_exposure")}})
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSPY buy & hold: CAGR {bench['cagr']:.1%}, max DD {bench['max_drawdown']:.1%}, "
          f"Sharpe {bench['sharpe']:.2f}")
    if args.out:
        df.to_csv(Path(args.out) / "compare.csv", index=False)


def cmd_walkforward(args):
    bars, spy = _load(args)
    base = Params(entry=args.entry, initial_equity=args.equity)
    cands = grid(base, {"init_stop_atr": [1.5, 2.0, 3.0], "trail_stop_atr": [2.0, 3.0, 4.0],
                        "partial_target_r": [0, 2.0]})
    oos, windows = walk_forward(bars, spy, cands, start=args.start)
    for w in windows:
        print(f"  test {w['test']}: {w['test_return']:+.1%} on {w['test_trades']} trades, "
              f"best {w['best']}")
    if len(oos):
        trades = pd.concat([w["trades"] for w in windows if len(w["trades"])], ignore_index=True)
        s = summarize(oos, trades)
        bench = summarize(buy_and_hold(spy.loc[oos.index], base.initial_equity))
        print(f"\nOut-of-sample, stitched\n{_fmt(s)}\n  SPY CAGR {bench['cagr']:.1%}, "
              f"max DD {bench['max_drawdown']:.1%}")
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True)
            oos.to_csv(Path(args.out) / f"walkforward_{args.entry}_equity.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trendbt")
    ap.add_argument("cmd", choices=["fetch", "backtest", "compare", "walkforward", "demo"])
    ap.add_argument("--cache", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--symbols", help="comma-separated; default is every US stock on Alpaca")
    ap.add_argument("--entry", default="breakout",
                    choices=["breakout", "pullback", "any", "momentum"])
    ap.add_argument("--equity", type=float, default=25_000)
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        cmd_fetch(args)
    elif args.cmd == "demo":
        args.out = None
        cmd_compare(args)
    else:
        {"backtest": cmd_backtest, "compare": cmd_compare,
         "walkforward": cmd_walkforward}[args.cmd](args)


if __name__ == "__main__":
    main()
