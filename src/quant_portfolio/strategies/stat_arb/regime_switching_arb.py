"""Regime Switching Arbitrage Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class RegimeSwitchingArb(Strategy):
    """Regime-dependent spread trading strategy.

    Identifies volatility regimes and trades mean reversion
    only during low-volatility (stable) regimes.

    Parameters
    ----------
    vol_window : int
        Window for volatility estimation (default 20).
    mr_window : int
        Window for mean reversion signal (default 30).
    entry_z : float
        Z-score entry threshold (default 1.5).
    """

    name = "regime_switching_arb"

    def __init__(self, vol_window: int = 20, mr_window: int = 30,
                 entry_z: float = 1.5, **kwargs):
        super().__init__(vol_window=vol_window, mr_window=mr_window,
                         entry_z=entry_z, **kwargs)
        self.vol_window = vol_window
        self.mr_window = mr_window
        self.entry_z = entry_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate regime-switching arbitrage signals."""
        close = data["close"]
        returns = close.pct_change()
        vol = returns.rolling(self.vol_window).std()
        vol_ma = vol.rolling(60).mean()
        low_vol_regime = vol < vol_ma

        ma = close.rolling(self.mr_window).mean()
        std = close.rolling(self.mr_window).std()
        zscore = (close - ma) / (std + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[low_vol_regime & (zscore < -self.entry_z)] = 1
        signal[low_vol_regime & (zscore > self.entry_z)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
