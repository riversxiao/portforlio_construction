"""Volatility Surface Strategy - term structure."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class VolSurface(Strategy):
    """Volatility term structure strategy.

    Uses the ratio of short-term to long-term realized volatility
    as a proxy for the vol term structure.

    Parameters
    ----------
    short_window : int
        Short-term vol window (default 5).
    long_window : int
        Long-term vol window (default 60).
    threshold : float
        Term structure ratio threshold (default 0.3).
    """

    name = "vol_surface"

    def __init__(self, short_window: int = 5, long_window: int = 60,
                 threshold: float = 0.3, **kwargs):
        super().__init__(short_window=short_window, long_window=long_window,
                         threshold=threshold, **kwargs)
        self.short_window = short_window
        self.long_window = long_window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate vol surface signals."""
        close = data["close"]
        returns = close.pct_change()
        short_vol = returns.rolling(self.short_window).std()
        long_vol = returns.rolling(self.long_window).std()

        term_structure = (short_vol / (long_vol + 1e-10)) - 1.0

        signal = pd.Series(0, index=data.index)
        # Contango (short < long): bullish
        signal[term_structure < -self.threshold] = 1
        # Backwardation (short > long): bearish
        signal[term_structure > self.threshold] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
