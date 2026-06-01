"""Volatility Mean Reversion Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class VolMeanReversion(Strategy):
    """Volatility mean reversion strategy.

    Trades expecting volatility to revert to its long-term mean.
    Buys after vol spikes (expecting normalization and price recovery).

    Parameters
    ----------
    vol_window : int
        Short vol window (default 10).
    avg_window : int
        Long-term vol average (default 60).
    entry_z : float
        Z-score threshold (default 1.5).
    """

    name = "vol_mean_reversion"

    def __init__(self, vol_window: int = 10, avg_window: int = 60,
                 entry_z: float = 1.5, **kwargs):
        super().__init__(vol_window=vol_window, avg_window=avg_window,
                         entry_z=entry_z, **kwargs)
        self.vol_window = vol_window
        self.avg_window = avg_window
        self.entry_z = entry_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate vol mean reversion signals."""
        close = data["close"]
        returns = close.pct_change()
        vol = returns.rolling(self.vol_window).std()
        vol_mean = vol.rolling(self.avg_window).mean()
        vol_std = vol.rolling(self.avg_window).std()
        vol_z = (vol - vol_mean) / (vol_std + 1e-10)

        signal = pd.Series(0, index=data.index)
        # Vol spike: expect reversion, buy the dip
        signal[vol_z > self.entry_z] = 1
        # Vol collapse: expect increase, reduce exposure
        signal[vol_z < -self.entry_z] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
