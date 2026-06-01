"""Beta Factor Strategy - betting against beta."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class BetaFactor(Strategy):
    """Betting against beta factor strategy.

    Buys low-beta assets and sells high-beta assets based on the
    empirical finding that low-beta assets are underpriced.

    Parameters
    ----------
    window : int
        Window for beta estimation (default 60).
    low_threshold : float
        Low beta threshold (default 0.7).
    high_threshold : float
        High beta threshold (default 1.3).
    """

    name = "beta_factor"

    def __init__(self, window: int = 60, low_threshold: float = 0.7,
                 high_threshold: float = 1.3, **kwargs):
        super().__init__(window=window, low_threshold=low_threshold,
                         high_threshold=high_threshold, **kwargs)
        self.window = window
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate betting against beta signals."""
        close = data["close"]
        returns = close.pct_change()
        market = returns.rolling(20).mean()

        rolling_cov = returns.rolling(self.window).cov(market)
        rolling_var = market.rolling(self.window).var()
        beta = rolling_cov / (rolling_var + 1e-10)
        beta = beta.clip(-5, 5)

        signal = pd.Series(0, index=data.index)
        signal[beta < self.low_threshold] = 1
        signal[beta > self.high_threshold] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
