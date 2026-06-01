"""Mean Field Arbitrage Strategy - cross-sectional mean-field model."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class MeanFieldArb(Strategy):
    """Mean-field statistical arbitrage strategy.

    Compares asset behavior to a rolling cross-sectional mean
    and trades deviations.

    Parameters
    ----------
    window : int
        Rolling window (default 30).
    entry_z : float
        Entry z-score threshold (default 2.0).
    """

    name = "mean_field_arb"

    def __init__(self, window: int = 30, entry_z: float = 2.0, **kwargs):
        super().__init__(window=window, entry_z=entry_z, **kwargs)
        self.window = window
        self.entry_z = entry_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean-field arbitrage signals."""
        close = data["close"]
        returns = close.pct_change()
        rolling_mean = returns.rolling(self.window).mean()
        rolling_std = returns.rolling(self.window).std()
        zscore = (returns - rolling_mean) / (rolling_std + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[zscore < -self.entry_z] = 1
        signal[zscore > self.entry_z] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
