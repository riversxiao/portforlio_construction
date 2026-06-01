"""Stochastic Oscillator Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class StochasticOscillator(Strategy):
    """Stochastic %K/%D oscillator strategy.

    Generates signals on %K/%D crossovers in oversold/overbought zones.

    Parameters
    ----------
    k_period : int
        %K period (default 14).
    d_period : int
        %D smoothing period (default 3).
    overbought : float
        Overbought level (default 80).
    oversold : float
        Oversold level (default 20).
    """

    name = "stochastic_oscillator"

    def __init__(self, k_period: int = 14, d_period: int = 3,
                 overbought: float = 80, oversold: float = 20, **kwargs):
        super().__init__(k_period=k_period, d_period=d_period,
                         overbought=overbought, oversold=oversold, **kwargs)
        self.k_period = k_period
        self.d_period = d_period
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate stochastic oscillator signals."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        lowest = low.rolling(self.k_period).min()
        highest = high.rolling(self.k_period).max()
        k = 100 * (close - lowest) / (highest - lowest + 1e-10)
        d = k.rolling(self.d_period).mean()

        signal = pd.Series(0, index=data.index)
        signal[(k > d) & (k < self.oversold)] = 1
        signal[(k < d) & (k > self.overbought)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
