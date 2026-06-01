"""Accruals Factor Strategy - accruals anomaly proxy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class AccrualsFactor(Strategy):
    """Accruals anomaly strategy using volume-price divergence.

    Uses divergence between price and volume trends as a proxy for
    earnings quality / accruals.

    Parameters
    ----------
    window : int
        Window for divergence calculation (default 60).
    threshold : float
        Divergence threshold (default 0.3).
    """

    name = "accruals_factor"

    def __init__(self, window: int = 60, threshold: float = 0.3, **kwargs):
        super().__init__(window=window, threshold=threshold, **kwargs)
        self.window = window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate accruals factor signals."""
        close = data["close"]
        volume = data["volume"]

        price_trend = close.pct_change(self.window)
        vol_trend = volume.rolling(self.window).mean().pct_change(self.window)

        # Divergence: price up but volume down suggests poor quality
        divergence = price_trend - vol_trend

        signal = pd.Series(0, index=data.index)
        signal[divergence < -self.threshold] = 1  # Volume confirms
        signal[divergence > self.threshold] = -1  # Divergence warning

        return pd.DataFrame({"signal": signal}, index=data.index)
