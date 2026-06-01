"""Microstructure Arbitrage Strategy - order flow imbalance signals."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class MicrostructureArb(Strategy):
    """Microstructure-based arbitrage using volume imbalance.

    Uses volume-price relationship to infer order flow imbalance
    and generates signals from flow direction.

    Parameters
    ----------
    window : int
        Rolling window for imbalance calculation (default 10).
    threshold : float
        Imbalance threshold for signal (default 0.6).
    """

    name = "microstructure_arb"

    def __init__(self, window: int = 10, threshold: float = 0.6, **kwargs):
        super().__init__(window=window, threshold=threshold, **kwargs)
        self.window = window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate microstructure arbitrage signals."""
        close = data["close"]
        volume = data["volume"]
        direction = np.sign(close.diff())
        buy_vol = volume * (direction == 1).astype(float)
        sell_vol = volume * (direction == -1).astype(float)

        buy_sum = buy_vol.rolling(self.window).sum()
        sell_sum = sell_vol.rolling(self.window).sum()
        total = buy_sum + sell_sum + 1e-10
        imbalance = (buy_sum - sell_sum) / total

        signal = pd.Series(0, index=data.index)
        signal[imbalance > self.threshold] = 1
        signal[imbalance < -self.threshold] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
