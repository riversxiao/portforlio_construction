"""ADF-Based Mean Reversion Strategy - Augmented Dickey-Fuller test signals."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class ADFBased(Strategy):
    """ADF test-based mean reversion strategy.

    Uses rolling ADF-like stationarity test (via OLS regression on lagged
    differences) to confirm mean-reverting behavior before trading.

    Parameters
    ----------
    window : int
        Rolling window for stationarity test (default 60).
    entry_z : float
        Z-score entry threshold (default 2.0).
    """

    name = "adf_based"

    def __init__(self, window: int = 60, entry_z: float = 2.0, **kwargs):
        super().__init__(window=window, entry_z=entry_z, **kwargs)
        self.window = window
        self.entry_z = entry_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on rolling ADF-like test."""
        close = data["close"]
        log_price = np.log(close)

        signal = pd.Series(0, index=data.index)
        for i in range(self.window, len(data)):
            window_data = log_price.iloc[i - self.window:i + 1].values
            y = np.diff(window_data)
            x = window_data[:-1]
            if len(x) < 5:
                continue
            x_dm = x - x.mean()
            beta = np.sum(x_dm * y) / (np.sum(x_dm ** 2) + 1e-10)
            if beta >= 0:
                continue  # Not mean-reverting

            mu = window_data.mean()
            std = window_data.std()
            if std < 1e-10:
                continue
            z = (window_data[-1] - mu) / std
            if z < -self.entry_z:
                signal.iloc[i] = 1
            elif z > self.entry_z:
                signal.iloc[i] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
