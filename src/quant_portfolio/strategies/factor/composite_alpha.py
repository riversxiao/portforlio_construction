"""Composite Alpha Strategy - alpha combination."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class CompositeAlpha(Strategy):
    """Composite alpha combination strategy.

    Combines momentum, mean reversion, and trend signals with
    adaptive weighting.

    Parameters
    ----------
    mom_window : int
        Momentum window (default 20).
    mr_window : int
        Mean reversion window (default 5).
    trend_window : int
        Trend window (default 50).
    """

    name = "composite_alpha"

    def __init__(self, mom_window: int = 20, mr_window: int = 5,
                 trend_window: int = 50, **kwargs):
        super().__init__(mom_window=mom_window, mr_window=mr_window,
                         trend_window=trend_window, **kwargs)
        self.mom_window = mom_window
        self.mr_window = mr_window
        self.trend_window = trend_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate composite alpha signals."""
        close = data["close"]

        # Momentum alpha
        mom = np.sign(close.pct_change(self.mom_window))

        # Mean reversion alpha
        mr_ma = close.rolling(self.mr_window).mean()
        mr_std = close.rolling(self.mr_window).std()
        mr_z = (close - mr_ma) / (mr_std + 1e-10)
        mr_signal = pd.Series(0.0, index=data.index)
        mr_signal[mr_z < -1.5] = 1
        mr_signal[mr_z > 1.5] = -1

        # Trend alpha
        trend_ma = close.rolling(self.trend_window).mean()
        trend_signal = np.sign(close - trend_ma)

        composite = (mom + mr_signal + trend_signal) / 3
        signal = pd.Series(0, index=data.index)
        signal[composite > 0.3] = 1
        signal[composite < -0.3] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
