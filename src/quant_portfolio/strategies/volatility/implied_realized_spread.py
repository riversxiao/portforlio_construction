"""Implied vs Realized Volatility Spread Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class ImpliedRealizedSpread(Strategy):
    """Implied vs realized volatility spread strategy.

    Uses short-term vs long-term realized vol as proxy for
    implied-realized spread.

    Parameters
    ----------
    short_window : int
        Short-term vol window (default 5).
    long_window : int
        Long-term vol window (default 30).
    threshold : float
        Spread threshold (default 0.5).
    """

    name = "implied_realized_spread"

    def __init__(self, short_window: int = 5, long_window: int = 30,
                 threshold: float = 0.5, **kwargs):
        super().__init__(short_window=short_window, long_window=long_window,
                         threshold=threshold, **kwargs)
        self.short_window = short_window
        self.long_window = long_window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals from vol spread."""
        close = data["close"]
        returns = close.pct_change()
        short_vol = returns.rolling(self.short_window).std() * np.sqrt(252)
        long_vol = returns.rolling(self.long_window).std() * np.sqrt(252)

        spread = (short_vol - long_vol) / (long_vol + 1e-10)

        signal = pd.Series(0, index=data.index)
        # High spread means elevated short-term vol -> sell
        signal[spread > self.threshold] = -1
        signal[spread < -self.threshold] = 1

        return pd.DataFrame({"signal": signal}, index=data.index)
