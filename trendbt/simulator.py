"""Day-by-day portfolio simulator.

Timeline for each trading day t:
  1. Open:  fill exits and entries decided at t-1's close (price = open +/- slippage).
  2. Intraday: check stops against the low (gap below the stop fills at the open),
     then the partial profit target against the high. When both could have been
     hit on the same bar the stop is assumed to come first (the conservative case).
  3. Close: ratchet trailing stops, apply breakeven / trend-break / time-stop
     rules, mark equity to market, and pick tomorrow's entries from today's signals.
"""
from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from .config import Params
from .signals import Prepared


@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop: float
    risk_per_share: float
    highest_close: float
    days_held: int = 0
    partial_done: bool = False
    exit_next_open: str | None = None   # reason, when a close-based rule fired
    realized: float = 0.0                # P&L already booked from partial sells
    cost_basis: float = 0.0


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame
    exposure: pd.Series
    params: Params
    info: dict = field(default_factory=dict)


def run(prep: Prepared, p: Params, start=None, end=None) -> Result:
    dates = prep.close.index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]

    cols = list(prep.close.columns)
    col_ix = {s: i for i, s in enumerate(cols)}
    full_ix = {d: i for i, d in enumerate(prep.close.index)}
    O, H, L, C = (prep.open.to_numpy(float), prep.high.to_numpy(float),
                  prep.low.to_numpy(float), prep.close.to_numpy(float))
    ATR = prep.atr.to_numpy(float)
    XMA = prep.exit_ma.to_numpy(float) if prep.exit_ma is not None else None
    SIG = prep.entry.to_numpy(bool)
    RANK = prep.rank.to_numpy(float)

    slip, comm = p.slippage, p.commission
    cash = p.initial_equity
    positions: dict[str, Position] = {}
    last_close: dict[str, float] = {}
    pending_entries: list[str] = []
    trades = []
    eq_curve, expo = [], []
    prev_equity = cash

    def close_out(pos: Position, price: float, date, reason: str, shares=None):
        nonlocal cash
        n = pos.shares if shares is None else shares
        fill = price * (1 - slip)
        cash += n * fill - comm
        pnl = n * (fill - pos.entry_price) - comm
        pos.realized += pnl
        pos.shares -= n
        if pos.shares == 0:
            trades.append({
                "symbol": pos.symbol, "entry_date": pos.entry_date,
                "exit_date": date, "entry_price": pos.entry_price,
                "exit_price": fill, "shares": round(pos.cost_basis / pos.entry_price),
                "pnl": pos.realized,
                "return": pos.realized / pos.cost_basis,
                "r_multiple": pos.realized / (pos.risk_per_share * pos.cost_basis / pos.entry_price),
                "days": pos.days_held, "reason": reason,
            })
            del positions[pos.symbol]

    for d in dates:
        t = full_ix[d]

        # 1. open: queued exits, then queued entries
        for pos in list(positions.values()):
            if pos.exit_next_open:
                j = col_ix[pos.symbol]
                px = O[t, j] if not math.isnan(O[t, j]) else last_close[pos.symbol]
                close_out(pos, px, d, pos.exit_next_open)

        for sym in pending_entries:
            if sym in positions or len(positions) >= p.max_positions:
                continue
            j = col_ix[sym]
            a = ATR[t - 1, j] if t > 0 else np.nan
            if math.isnan(O[t, j]) or math.isnan(a) or a <= 0:
                continue
            entry = O[t, j] * (1 + slip)
            stop = entry - p.init_stop_atr * a
            rps = entry - stop
            shares = int(min(p.risk_per_trade * prev_equity / rps,
                             p.max_position_pct * prev_equity / entry,
                             (cash - comm) / entry))
            if shares < 1:
                continue
            cash -= shares * entry + comm
            positions[sym] = Position(sym, d, entry, shares, stop, rps,
                                      highest_close=entry, cost_basis=shares * entry)
        pending_entries = []

        # 2. intraday: stops first, then partial target
        for pos in list(positions.values()):
            j = col_ix[pos.symbol]
            o, h, l = O[t, j], H[t, j], L[t, j]
            if math.isnan(o):
                continue  # no bar today (halt or delisting); hold at last close
            if o <= pos.stop:
                close_out(pos, o, d, "stop_gap")
                continue
            if l <= pos.stop:
                close_out(pos, pos.stop, d, "stop")
                continue
            if p.partial_target_r and not pos.partial_done:
                target = pos.entry_price + p.partial_target_r * pos.risk_per_share
                if h >= target and pos.shares >= 2:
                    close_out(pos, max(o, target), d, "target", shares=pos.shares // 2)
                    pos.partial_done = True

        # 3. close: update stops and close-based rules, mark to market
        for pos in positions.values():
            j = col_ix[pos.symbol]
            c = C[t, j]
            if math.isnan(c):
                continue
            last_close[pos.symbol] = c
            pos.days_held += 1
            pos.highest_close = max(pos.highest_close, c)
            a = ATR[t, j]
            if not math.isnan(a):
                pos.stop = max(pos.stop, pos.highest_close - p.trail_stop_atr * a)
            if p.breakeven_r and c >= pos.entry_price + p.breakeven_r * pos.risk_per_share:
                pos.stop = max(pos.stop, pos.entry_price)
            if XMA is not None and not math.isnan(XMA[t, j]) and c < XMA[t, j]:
                pos.exit_next_open = "trend_break"
            elif (p.time_stop_days and pos.days_held >= p.time_stop_days
                  and c < pos.entry_price + pos.risk_per_share and not pos.partial_done):
                pos.exit_next_open = "time_stop"

        held_value = sum(pos.shares * last_close.get(pos.symbol, pos.entry_price)
                         for pos in positions.values())
        equity = cash + held_value
        eq_curve.append(equity)
        expo.append(held_value / equity if equity > 0 else 0.0)
        prev_equity = equity

        # tomorrow's entries, best relative strength first
        open_after_exits = sum(1 for pos in positions.values() if not pos.exit_next_open)
        slots = p.max_positions - open_after_exits
        if slots > 0:
            cand = np.flatnonzero(SIG[t])
            cand = [j for j in cand if cols[j] not in positions]
            cand.sort(key=lambda j: -np.nan_to_num(RANK[t, j], nan=-np.inf))
            pending_entries = [cols[j] for j in cand[:slots]]

    # liquidate at the last available close so every trade is counted
    if len(dates):
        for pos in list(positions.values()):
            close_out(pos, last_close.get(pos.symbol, pos.entry_price),
                      dates[-1], "end_of_test")

    return Result(
        equity=pd.Series(eq_curve, index=dates, name="equity"),
        trades=pd.DataFrame(trades),
        exposure=pd.Series(expo, index=dates, name="exposure"),
        params=p,
    )
