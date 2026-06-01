"""Liquidity Factor Strategy - illiquidity premium."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class LiquidityFactor(Strategy):
    """Liquidity factor strategy using Amihud illiquidity measure.

    Uses price impact (return/volume) as illiquidity proxy.
    Illiquid assets earn a premium.

    Parameters
    ----------
    window : int
        Window for illiquidity calculation (default 20).
    threshold : float
        Illiquidity percentile threshold (default 0.7).
    """

    name = "liquidity_factor"

    def __init__(self, window: int = 20, threshold: float = 0.7, **kwargs):
        super().__init__(window=window, threshold=threshold, **kwargs)
        self.window = window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate liquidity factor signals."""
        close = data["close"]
        volume = data["volume"]
        returns = close.pct_change().abs()
        illiquidity = returns / (volume + 1e-10)
        avg_illiq = illiquidity.rolling(self.window).mean()
        illiq_rank = avg_illiq.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        mom = close.pct_change(self.window)
        signal = pd.Series(0, index=data.index)
        signal[(illiq_rank > self.threshold) & (mom > 0)] = 1
        signal[(illiq_rank < 1 - self.threshold) & (mom < 0)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
