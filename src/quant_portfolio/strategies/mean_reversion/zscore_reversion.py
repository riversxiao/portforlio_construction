"""Z-Score Reversion Strategy - z-score threshold trading."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class ZScoreReversion(Strategy):
    """Z-score mean reversion strategy.

    Computes rolling z-score of price and generates signals when the
    z-score exceeds thresholds, expecting reversion to the mean.

    Parameters
    ----------
    window : int
        Rolling window for z-score calculation (default 20).
    entry_z : float
        Z-score entry threshold (default 2.0).
    exit_z : float
        Z-score exit threshold (default 0.0).
    """

    name = "zscore_reversion"

    def __init__(self, window: int = 20, entry_z: float = 2.0,
                 exit_z: float = 0.0, **kwargs):
        super().__init__(window=window, entry_z=entry_z, exit_z=exit_z, **kwargs)
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on rolling z-score thresholds."""
        close = data["close"]
        ma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        zscore = (close - ma) / (std + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[zscore < -self.entry_z] = 1
        signal[zscore > self.entry_z] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
