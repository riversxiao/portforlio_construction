"""Time Series Momentum Strategy - TSMOM factor."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class TimeSeriesMomentum(Strategy):
    """Time-series momentum (TSMOM) strategy.

    Generates signals based on the sign of past returns scaled by
    realized volatility, following Moskowitz et al. (2012).

    Parameters
    ----------
    lookback : int
        Return lookback period (default 252).
    vol_lookback : int
        Volatility estimation period (default 60).
    """

    name = "time_series_momentum"

    def __init__(self, lookback: int = 252, vol_lookback: int = 60, **kwargs):
        super().__init__(lookback=lookback, vol_lookback=vol_lookback, **kwargs)
        self.lookback = lookback
        self.vol_lookback = vol_lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate TSMOM signals."""
        close = data["close"]
        returns = close.pct_change()
        cum_return = close.pct_change(self.lookback)
        vol = returns.rolling(self.vol_lookback).std() * np.sqrt(252)

        risk_adj_mom = cum_return / (vol + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[risk_adj_mom > 0] = 1
        signal[risk_adj_mom < 0] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
