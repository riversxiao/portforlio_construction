"""Value Factor Strategy - book-to-market value proxy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class ValueFactor(Strategy):
    """Value factor strategy using price-based value proxy.

    Uses price-to-moving-average ratio as a value proxy. Lower ratio
    suggests undervaluation.

    Parameters
    ----------
    lookback : int
        Lookback for long-term average (default 252).
    threshold : float
        Value threshold (default 0.8).
    """

    name = "value_factor"

    def __init__(self, lookback: int = 252, threshold: float = 0.8, **kwargs):
        super().__init__(lookback=lookback, threshold=threshold, **kwargs)
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate value factor signals."""
        close = data["close"]
        long_ma = close.rolling(self.lookback).mean()
        value_ratio = close / (long_ma + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[value_ratio < self.threshold] = 1
        signal[value_ratio > 1.0 / self.threshold] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
