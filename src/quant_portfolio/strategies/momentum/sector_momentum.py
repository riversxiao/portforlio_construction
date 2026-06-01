"""Sector Momentum Strategy - rotation based on sector momentum rankings."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class SectorMomentum(Strategy):
    """Sector momentum rotation strategy.

    Uses rolling returns to rank assets and generates buy signals for
    top performers and sell signals for bottom performers.

    Parameters
    ----------
    lookback : int
        Lookback period for return calculation (default 60).
    top_pct : float
        Top percentile threshold for buy signals (default 0.7).
    """

    name = "sector_momentum"

    def __init__(self, lookback: int = 60, top_pct: float = 0.7, **kwargs):
        super().__init__(lookback=lookback, top_pct=top_pct, **kwargs)
        self.lookback = lookback
        self.top_pct = top_pct

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on sector-style momentum ranking."""
        close = data["close"]
        returns = close.pct_change(self.lookback)
        rolling_rank = returns.rolling(self.lookback).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-10), raw=False
        )

        signal = pd.Series(0, index=data.index)
        signal[rolling_rank > 1.0] = 1
        signal[rolling_rank < -1.0] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
