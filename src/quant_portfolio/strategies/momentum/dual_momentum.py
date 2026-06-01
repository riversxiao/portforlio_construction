"""Dual Momentum Strategy - combines absolute and relative momentum."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class DualMomentum(Strategy):
    """Dual momentum strategy combining absolute and relative momentum.

    Buy when both absolute momentum (return > 0) and relative momentum
    (return > benchmark/risk-free proxy) are positive. Sell when both negative.

    Parameters
    ----------
    lookback : int
        Lookback period for momentum calculation (default 252).
    short_lookback : int
        Short-term lookback for relative comparison (default 63).
    """

    name = "dual_momentum"

    def __init__(self, lookback: int = 252, short_lookback: int = 63, **kwargs):
        super().__init__(lookback=lookback, short_lookback=short_lookback, **kwargs)
        self.lookback = lookback
        self.short_lookback = short_lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on dual momentum criteria."""
        close = data["close"]
        abs_mom = close.pct_change(self.lookback)
        rel_mom = close.pct_change(self.short_lookback)

        signal = pd.Series(0, index=data.index)
        signal[(abs_mom > 0) & (rel_mom > 0)] = 1
        signal[(abs_mom < 0) & (rel_mom < 0)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
