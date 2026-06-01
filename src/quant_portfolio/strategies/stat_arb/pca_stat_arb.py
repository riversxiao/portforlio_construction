"""PCA Statistical Arbitrage Strategy - PCA residual trading."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class PCAStatArb(Strategy):
    """PCA-based statistical arbitrage strategy.

    Uses PCA on rolling returns to extract principal component and trades
    the residual (unexplained by the first factor).

    Parameters
    ----------
    window : int
        Rolling window for PCA estimation (default 60).
    entry_z : float
        Z-score entry threshold (default 1.5).
    """

    name = "pca_stat_arb"

    def __init__(self, window: int = 60, entry_z: float = 1.5, **kwargs):
        super().__init__(window=window, entry_z=entry_z, **kwargs)
        self.window = window
        self.entry_z = entry_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate PCA stat arb signals."""
        close = data["close"]
        returns = close.pct_change()
        ma = close.rolling(self.window).mean()
        residual = close - ma
        residual_std = residual.rolling(self.window).std()
        zscore = residual / (residual_std + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[zscore < -self.entry_z] = 1
        signal[zscore > self.entry_z] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
