"""Volatility Risk Premium Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class VolRiskPremium(Strategy):
    """Volatility risk premium harvesting strategy.

    Sells volatility (go long) when vol is elevated above historical
    average, capturing the vol risk premium.

    Parameters
    ----------
    vol_window : int
        Current vol window (default 20).
    avg_window : int
        Historical average window (default 120).
    premium_threshold : float
        Premium threshold multiplier (default 1.2).
    """

    name = "vol_risk_premium"

    def __init__(self, vol_window: int = 20, avg_window: int = 120,
                 premium_threshold: float = 1.2, **kwargs):
        super().__init__(vol_window=vol_window, avg_window=avg_window,
                         premium_threshold=premium_threshold, **kwargs)
        self.vol_window = vol_window
        self.avg_window = avg_window
        self.premium_threshold = premium_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate vol risk premium signals."""
        close = data["close"]
        returns = close.pct_change()
        current_vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        avg_vol = current_vol.rolling(self.avg_window).mean()

        ratio = current_vol / (avg_vol + 1e-10)

        signal = pd.Series(0, index=data.index)
        # Sell vol (go long) when current vol > average (premium exists)
        signal[ratio > self.premium_threshold] = 1
        signal[ratio < 1.0 / self.premium_threshold] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
