"""Investment Factor Strategy - conservative investment."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class InvestmentFactor(Strategy):
    """Conservative investment factor strategy.

    Prefers assets with low asset growth (conservative investment).
    Uses price stability as a proxy.

    Parameters
    ----------
    window : int
        Lookback window (default 252).
    threshold : float
        Growth threshold (default 0.3).
    """

    name = "investment_factor"

    def __init__(self, window: int = 252, threshold: float = 0.3, **kwargs):
        super().__init__(window=window, threshold=threshold, **kwargs)
        self.window = window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate investment factor signals."""
        close = data["close"]
        annual_return = close.pct_change(self.window)
        vol = close.pct_change().rolling(self.window).std() * np.sqrt(252)

        # Conservative: moderate growth with low vol
        signal = pd.Series(0, index=data.index)
        signal[(annual_return > 0) & (annual_return < self.threshold) & (vol < 0.25)] = 1
        signal[(annual_return > self.threshold * 2) & (vol > 0.4)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
