"""Momentum Reversal Strategy - short-term reversal with long-term momentum."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class MomentumReversal(Strategy):
    """Momentum reversal strategy.

    Combines long-term momentum with short-term mean reversion.
    Buys when long-term trend is positive but short-term has pulled back.

    Parameters
    ----------
    long_period : int
        Long-term momentum lookback (default 252).
    short_period : int
        Short-term reversal lookback (default 5).
    reversal_threshold : float
        Short-term drop threshold for entry (default -0.03).
    """

    name = "momentum_reversal"

    def __init__(self, long_period: int = 252, short_period: int = 5,
                 reversal_threshold: float = -0.03, **kwargs):
        super().__init__(long_period=long_period, short_period=short_period,
                         reversal_threshold=reversal_threshold, **kwargs)
        self.long_period = long_period
        self.short_period = short_period
        self.reversal_threshold = reversal_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals combining momentum and reversal."""
        close = data["close"]
        long_mom = close.pct_change(self.long_period)
        short_ret = close.pct_change(self.short_period)

        signal = pd.Series(0, index=data.index)
        signal[(long_mom > 0) & (short_ret < self.reversal_threshold)] = 1
        signal[(long_mom < 0) & (short_ret > -self.reversal_threshold)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
