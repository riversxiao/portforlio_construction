"""Ornstein-Uhlenbeck Mean Reversion Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class OrnsteinUhlenbeck(Strategy):
    """Ornstein-Uhlenbeck process-based mean reversion.

    Fits an OU process to log prices and trades based on deviation
    from the estimated mean level.

    Parameters
    ----------
    window : int
        Window for parameter estimation (default 60).
    entry_threshold : float
        Entry threshold in standard deviations (default 1.5).
    """

    name = "ornstein_uhlenbeck"

    def __init__(self, window: int = 60, entry_threshold: float = 1.5, **kwargs):
        super().__init__(window=window, entry_threshold=entry_threshold, **kwargs)
        self.window = window
        self.entry_threshold = entry_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on OU process fit."""
        close = data["close"]
        log_price = np.log(close)

        signal = pd.Series(0, index=data.index)
        for i in range(self.window, len(data)):
            window_data = log_price.iloc[i - self.window:i]
            mu = window_data.mean()
            sigma = window_data.std()
            if sigma < 1e-10:
                continue
            deviation = (log_price.iloc[i] - mu) / sigma
            if deviation < -self.entry_threshold:
                signal.iloc[i] = 1
            elif deviation > self.entry_threshold:
                signal.iloc[i] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
