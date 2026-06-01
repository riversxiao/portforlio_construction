"""Moving Average Reversion Strategy - distance from MA reversion."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class MovingAverageReversion(Strategy):
    """Moving average distance mean reversion strategy.

    Trades reversion when price deviates significantly from its
    moving average.

    Parameters
    ----------
    window : int
        Moving average window (default 50).
    threshold : float
        Deviation threshold as fraction (default 0.05).
    """

    name = "moving_average_reversion"

    def __init__(self, window: int = 50, threshold: float = 0.05, **kwargs):
        super().__init__(window=window, threshold=threshold, **kwargs)
        self.window = window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on distance from moving average."""
        close = data["close"]
        ma = close.rolling(self.window).mean()
        deviation = (close - ma) / (ma + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[deviation < -self.threshold] = 1
        signal[deviation > self.threshold] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
