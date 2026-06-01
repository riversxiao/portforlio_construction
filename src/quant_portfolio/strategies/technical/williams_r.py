"""Williams %R Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class WilliamsR(Strategy):
    """Williams %R oscillator strategy.

    Generates signals based on overbought/oversold Williams %R readings.

    Parameters
    ----------
    period : int
        Lookback period (default 14).
    overbought : float
        Overbought level (default -20).
    oversold : float
        Oversold level (default -80).
    """

    name = "williams_r"

    def __init__(self, period: int = 14, overbought: float = -20,
                 oversold: float = -80, **kwargs):
        super().__init__(period=period, overbought=overbought, oversold=oversold, **kwargs)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Williams %R signals."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        highest = high.rolling(self.period).max()
        lowest = low.rolling(self.period).min()
        wr = -100 * (highest - close) / (highest - lowest + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[wr < self.oversold] = 1
        signal[wr > self.overbought] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
