"""Variance Ratio Strategy - variance ratio test signals."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class VarianceRatio(Strategy):
    """Variance ratio test-based mean reversion strategy.

    Uses the variance ratio test to detect mean reversion and generates
    signals when the ratio indicates reversion and price deviates.

    Parameters
    ----------
    window : int
        Window for variance ratio calculation (default 60).
    holding_period : int
        Holding period for ratio calculation (default 5).
    entry_z : float
        Z-score entry threshold (default 1.5).
    """

    name = "variance_ratio"

    def __init__(self, window: int = 60, holding_period: int = 5,
                 entry_z: float = 1.5, **kwargs):
        super().__init__(window=window, holding_period=holding_period,
                         entry_z=entry_z, **kwargs)
        self.window = window
        self.holding_period = holding_period
        self.entry_z = entry_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on variance ratio test."""
        close = data["close"]
        returns = close.pct_change()

        signal = pd.Series(0, index=data.index)
        for i in range(self.window, len(data)):
            r = returns.iloc[i - self.window:i].dropna().values
            if len(r) < 20:
                continue
            var_1 = np.var(r)
            if var_1 < 1e-10:
                continue
            # Compute k-period returns
            k = self.holding_period
            r_k = np.array([r[j:j + k].sum() for j in range(0, len(r) - k, k)])
            if len(r_k) < 3:
                continue
            var_k = np.var(r_k)
            vr = var_k / (k * var_1)

            if vr < 0.8:  # Mean-reverting
                window_prices = close.iloc[i - self.window:i + 1].values
                mu = np.mean(window_prices)
                std = np.std(window_prices)
                if std < 1e-10:
                    continue
                z = (window_prices[-1] - mu) / std
                if z < -self.entry_z:
                    signal.iloc[i] = 1
                elif z > self.entry_z:
                    signal.iloc[i] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
