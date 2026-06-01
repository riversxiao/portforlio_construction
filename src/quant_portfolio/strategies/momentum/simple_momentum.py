"""Simple Momentum Strategy - price momentum over N days."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class SimpleMomentum(Strategy):
    """Simple price momentum strategy.

    Generates buy signals when the N-day return exceeds a threshold,
    sell signals when it falls below the negative threshold, and hold otherwise.

    Parameters
    ----------
    lookback : int
        Number of days to measure momentum (default 20).
    threshold : float
        Minimum return threshold for signal generation (default 0.02).
    """

    name = "simple_momentum"

    def __init__(self, lookback: int = 20, threshold: float = 0.02, **kwargs):
        super().__init__(lookback=lookback, threshold=threshold, **kwargs)
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on N-day price momentum."""
        close = data["close"]
        momentum = close.pct_change(self.lookback)

        signal = pd.Series(0, index=data.index)
        signal[momentum > self.threshold] = 1
        signal[momentum < -self.threshold] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
