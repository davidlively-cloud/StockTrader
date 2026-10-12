"""Performance metrics from an equity curve and a trade list."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min()) if len(equity) else 0.0


def summarize(equity: pd.Series, trades: pd.DataFrame | None = None,
              exposure: pd.Series | None = None) -> dict:
    out = {}
    if len(equity) < 2:
        return out
    years = len(equity) / TRADING_DAYS
    rets = equity.pct_change().dropna()
    out["start"] = equity.index[0].date().isoformat()
    out["end"] = equity.index[-1].date().isoformat()
    out["final_equity"] = float(equity.iloc[-1])
    out["total_return"] = float(equity.iloc[-1] / equity.iloc[0] - 1)
    out["cagr"] = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
    out["max_drawdown"] = max_drawdown(equity)
    sd = rets.std()
    out["sharpe"] = float(rets.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    if exposure is not None and len(exposure):
        out["avg_exposure"] = float(exposure.mean())
    if trades is not None and len(trades):
        wins = trades[trades.pnl > 0]
        losses = trades[trades.pnl <= 0]
        out["trades"] = int(len(trades))
        out["trades_per_year"] = float(len(trades) / years)
        out["win_rate"] = float(len(wins) / len(trades))
        out["avg_win_pct"] = float(wins["return"].mean()) if len(wins) else 0.0
        out["avg_loss_pct"] = float(losses["return"].mean()) if len(losses) else 0.0
        out["win_loss_ratio"] = (abs(out["avg_win_pct"] / out["avg_loss_pct"])
                                 if out["avg_loss_pct"] else float("inf"))
        gross_loss = -losses.pnl.sum()
        out["profit_factor"] = float(wins.pnl.sum() / gross_loss) if gross_loss > 0 else float("inf")
        out["avg_r"] = float(trades.r_multiple.mean())
        out["avg_days_held"] = float(trades.days.mean())
        out["exit_reasons"] = trades.reason.value_counts().to_dict()
    else:
        out["trades"] = 0
    return out


def buy_and_hold(close: pd.Series, initial: float) -> pd.Series:
    close = close.dropna()
    return initial * close / close.iloc[0]
