"""Volatility Momentum Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class VolMomentum(Strategy):
    """Volatility momentum strategy.

    Trades based on the direction of volatility change.
    Rising vol suggests selling, falling vol suggests buying.

    Parameters
    ----------
    vol_window : int
        Volatility window (default 20).
    change_period : int
        Period for vol change measurement (default 10).
    """

    name = "vol_momentum"

    def __init__(self, vol_window: int = 20, change_period: int = 10, **kwargs):
        super().__init__(vol_window=vol_window, change_period=change_period, **kwargs)
        self.vol_window = vol_window
        self.change_period = change_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate vol momentum signals."""
        close = data["close"]
        returns = close.pct_change()
        vol = returns.rolling(self.vol_window).std()
        vol_change = vol.pct_change(self.change_period)

        signal = pd.Series(0, index=data.index)
        signal[vol_change < -0.2] = 1  # Falling vol: bullish
        signal[vol_change > 0.2] = -1  # Rising vol: bearish

        return pd.DataFrame({"signal": signal}, index=data.index)
