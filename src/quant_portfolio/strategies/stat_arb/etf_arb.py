"""ETF Arbitrage Strategy - ETF vs constituents arbitrage."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class ETFArb(Strategy):
    """ETF vs synthetic NAV arbitrage strategy.

    Uses the deviation between the asset and its smoothed value (NAV proxy)
    to generate arbitrage signals.

    Parameters
    ----------
    nav_window : int
        Window for NAV proxy (smoothed price) (default 10).
    entry_threshold : float
        Entry threshold as fraction (default 0.01).
    """

    name = "etf_arb"

    def __init__(self, nav_window: int = 10, entry_threshold: float = 0.01, **kwargs):
        super().__init__(nav_window=nav_window, entry_threshold=entry_threshold, **kwargs)
        self.nav_window = nav_window
        self.entry_threshold = entry_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ETF arbitrage signals."""
        close = data["close"]
        nav_proxy = close.rolling(self.nav_window).mean()
        premium = (close - nav_proxy) / (nav_proxy + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[premium < -self.entry_threshold] = 1
        signal[premium > self.entry_threshold] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
