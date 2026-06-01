"""Bollinger Bands Mean Reversion Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class BollingerBands(Strategy):
    """Bollinger Bands mean reversion strategy.

    Buys when price touches/crosses below the lower band and sells
    when price touches/crosses above the upper band.

    Parameters
    ----------
    window : int
        Rolling window for moving average (default 20).
    num_std : float
        Number of standard deviations for bands (default 2.0).
    """

    name = "bollinger_bands"

    def __init__(self, window: int = 20, num_std: float = 2.0, **kwargs):
        super().__init__(window=window, num_std=num_std, **kwargs)
        self.window = window
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on Bollinger Band crossovers."""
        close = data["close"]
        ma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        upper = ma + self.num_std * std
        lower = ma - self.num_std * std

        signal = pd.Series(0, index=data.index)
        signal[close < lower] = 1
        signal[close > upper] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
