"""Mean Reversion Portfolio Strategy - portfolio-level mean reversion."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class MeanReversionPortfolio(Strategy):
    """Portfolio-level mean reversion strategy.

    Combines multiple lookbacks to generate a composite reversion signal.

    Parameters
    ----------
    short_window : int
        Short-term reversion window (default 5).
    medium_window : int
        Medium-term reversion window (default 20).
    long_window : int
        Long-term reversion window (default 60).
    """

    name = "mean_reversion_portfolio"

    def __init__(self, short_window: int = 5, medium_window: int = 20,
                 long_window: int = 60, **kwargs):
        super().__init__(short_window=short_window, medium_window=medium_window,
                         long_window=long_window, **kwargs)
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate composite mean reversion signals."""
        close = data["close"]

        def zscore(series, window):
            ma = series.rolling(window).mean()
            std = series.rolling(window).std()
            return (series - ma) / (std + 1e-10)

        z_short = zscore(close, self.short_window)
        z_medium = zscore(close, self.medium_window)
        z_long = zscore(close, self.long_window)

        composite = (z_short + z_medium + z_long) / 3

        signal = pd.Series(0, index=data.index)
        signal[composite < -1.5] = 1
        signal[composite > 1.5] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
