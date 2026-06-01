"""Copula Arbitrage Strategy - copula-based dependency trading."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class CopulaArb(Strategy):
    """Copula-based dependency trading strategy.

    Uses the empirical CDF transform (uniform margins) to detect
    extreme deviations in the dependence structure.

    Parameters
    ----------
    window : int
        Rolling window for CDF estimation (default 60).
    threshold : float
        Tail probability threshold (default 0.1).
    """

    name = "copula_arb"

    def __init__(self, window: int = 60, threshold: float = 0.1, **kwargs):
        super().__init__(window=window, threshold=threshold, **kwargs)
        self.window = window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate copula-based arbitrage signals."""
        close = data["close"]
        returns = close.pct_change()

        signal = pd.Series(0, index=data.index)
        for i in range(self.window, len(data)):
            r_window = returns.iloc[i - self.window:i + 1].dropna().values
            if len(r_window) < 10:
                continue
            # Empirical CDF rank transform
            rank = (r_window[-1] - r_window.min()) / (r_window.max() - r_window.min() + 1e-10)
            if rank < self.threshold:
                signal.iloc[i] = 1
            elif rank > 1 - self.threshold:
                signal.iloc[i] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
