"""Cointegration Basket Strategy - basket mean reversion."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class CointegrationBasket(Strategy):
    """Basket mean reversion strategy.

    Creates a synthetic mean-reverting basket using the spread between
    price and a weighted combination of multiple moving averages.

    Parameters
    ----------
    windows : list
        List of MA windows for basket construction (default [10, 20, 50]).
    entry_z : float
        Z-score entry threshold (default 2.0).
    lookback : int
        Lookback for z-score calculation (default 60).
    """

    name = "cointegration_basket"

    def __init__(self, windows: list = None, entry_z: float = 2.0,
                 lookback: int = 60, **kwargs):
        if windows is None:
            windows = [10, 20, 50]
        super().__init__(windows=windows, entry_z=entry_z,
                         lookback=lookback, **kwargs)
        self.windows = windows
        self.entry_z = entry_z
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate basket mean reversion signals."""
        close = data["close"]
        basket = sum(close.rolling(w).mean() for w in self.windows) / len(self.windows)
        spread = close - basket
        spread_mean = spread.rolling(self.lookback).mean()
        spread_std = spread.rolling(self.lookback).std()
        zscore = (spread - spread_mean) / (spread_std + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[zscore < -self.entry_z] = 1
        signal[zscore > self.entry_z] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
