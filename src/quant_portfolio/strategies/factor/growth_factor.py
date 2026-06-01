"""Growth Factor Strategy - earnings/revenue growth proxy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class GrowthFactor(Strategy):
    """Growth factor strategy using price momentum as growth proxy.

    Uses sustained positive momentum as a proxy for earnings growth.

    Parameters
    ----------
    short_window : int
        Short-term growth window (default 20).
    long_window : int
        Long-term growth window (default 60).
    """

    name = "growth_factor"

    def __init__(self, short_window: int = 20, long_window: int = 60, **kwargs):
        super().__init__(short_window=short_window, long_window=long_window, **kwargs)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate growth factor signals."""
        close = data["close"]
        short_ret = close.pct_change(self.short_window)
        long_ret = close.pct_change(self.long_window)

        signal = pd.Series(0, index=data.index)
        signal[(short_ret > 0) & (long_ret > 0) & (short_ret > long_ret / 3)] = 1
        signal[(short_ret < 0) & (long_ret < 0)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
