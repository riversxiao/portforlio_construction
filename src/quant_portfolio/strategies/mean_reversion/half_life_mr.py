"""Half-Life Mean Reversion Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class HalfLifeMR(Strategy):
    """Half-life based mean reversion strategy.

    Estimates the half-life of mean reversion using OLS on lag-1 spread
    and uses it for entry/exit timing.

    Parameters
    ----------
    lookback : int
        Lookback window for half-life estimation (default 120).
    entry_z : float
        Entry z-score threshold (default 1.5).
    """

    name = "half_life_mr"

    def __init__(self, lookback: int = 120, entry_z: float = 1.5, **kwargs):
        super().__init__(lookback=lookback, entry_z=entry_z, **kwargs)
        self.lookback = lookback
        self.entry_z = entry_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using half-life mean reversion."""
        close = data["close"]
        log_price = np.log(close)
        spread = log_price - log_price.rolling(self.lookback).mean()

        # Estimate half-life using lag-1 regression
        spread_lag = spread.shift(1)
        delta_spread = spread - spread_lag

        signal = pd.Series(0, index=data.index)
        for i in range(self.lookback + 1, len(data)):
            y = delta_spread.iloc[i - self.lookback:i].dropna().values
            x = spread_lag.iloc[i - self.lookback:i].dropna().values
            if len(x) < 10 or len(y) < 10:
                continue
            min_len = min(len(x), len(y))
            x, y = x[:min_len], y[:min_len]
            beta = np.sum(x * y) / (np.sum(x * x) + 1e-10)
            if beta >= 0:
                continue
            half_life = -np.log(2) / beta
            if half_life < 1 or half_life > self.lookback:
                continue

            std = spread.iloc[i - self.lookback:i].std()
            z = spread.iloc[i] / (std + 1e-10)
            if z < -self.entry_z:
                signal.iloc[i] = 1
            elif z > self.entry_z:
                signal.iloc[i] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
