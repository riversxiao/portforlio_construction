"""Cross-Sectional Momentum Strategy - ranking-based momentum."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class CrossSectionalMomentum(Strategy):
    """Cross-sectional momentum strategy.

    Ranks the asset's rolling return within its own historical distribution
    and generates signals based on the rank percentile.

    Parameters
    ----------
    lookback : int
        Return calculation period (default 60).
    rank_window : int
        Window for percentile rank calculation (default 252).
    """

    name = "cross_sectional_momentum"

    def __init__(self, lookback: int = 60, rank_window: int = 252, **kwargs):
        super().__init__(lookback=lookback, rank_window=rank_window, **kwargs)
        self.lookback = lookback
        self.rank_window = rank_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-sectional momentum signals."""
        close = data["close"]
        returns = close.pct_change(self.lookback)

        pct_rank = returns.rolling(self.rank_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        signal = pd.Series(0, index=data.index)
        signal[pct_rank > 0.7] = 1
        signal[pct_rank < 0.3] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
