"""Strategy and backtest settings. Defaults match the Stock Trend Prototype Plan."""
from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class Params:
    # Universe filter
    min_price: float = 10.0
    min_dollar_volume: float = 20e6      # 20-day average close * volume
    use_regime_filter: bool = True       # only buy when SPY > its 200-day average

    # Trend filter
    fast_ma: int = 50
    slow_ma: int = 200
    slope_lookback: int = 10             # fast MA must be higher than N days ago
    adx_period: int = 14
    min_adx: float = 20.0
    rs_lookback: int = 126               # ~6 months
    min_rs_pct: float = 0.80             # top 20% of the universe

    # Entry: "breakout", "pullback", "momentum", or "any" (breakout or pullback)
    entry: str = "breakout"
    breakout_days: int = 20
    pullback_ma: int = 20
    rsi_period: int = 2
    rsi_oversold: float = 10.0
    momentum_top_n: int = 10             # momentum mode: buy top N, re-ranked weekly

    # Exits
    atr_period: int = 14
    init_stop_atr: float = 2.0
    trail_stop_atr: float = 3.0          # chandelier: highest close - k * ATR
    trend_exit_ma: int = 0               # 0 = off; else exit on close below this MA
    partial_target_r: float = 2.0        # 0 = off; sell half at entry + k * R
    breakeven_r: float = 1.0             # 0 = off; raise stop to entry after +k * R
    time_stop_days: int = 10             # 0 = off; exit if not up 1R after N days

    # Portfolio
    initial_equity: float = 25_000.0
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.10
    max_positions: int = 10
    slippage: float = 0.0005             # per side, as a fraction of price
    commission: float = 0.0

    def to_dict(self):
        return asdict(self)

    def with_(self, **kw):
        return replace(self, **kw)
