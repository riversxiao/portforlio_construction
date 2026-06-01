"""Supertrend Strategy - ATR-based supertrend indicator."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class Supertrend(Strategy):
    """ATR-based Supertrend trend following strategy.

    Uses Average True Range to set dynamic support/resistance bands
    and generates signals on band crossovers.

    Parameters
    ----------
    period : int
        ATR period (default 10).
    multiplier : float
        ATR multiplier for band width (default 3.0).
    """

    name = "supertrend"

    def __init__(self, period: int = 10, multiplier: float = 3.0, **kwargs):
        super().__init__(period=period, multiplier=multiplier, **kwargs)
        self.period = period
        self.multiplier = multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Supertrend signals."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.period).mean()

        hl2 = (high + low) / 2
        upper_band = hl2 + self.multiplier * atr
        lower_band = hl2 - self.multiplier * atr

        supertrend = pd.Series(0.0, index=data.index)
        direction = pd.Series(1, index=data.index)

        for i in range(self.period, len(data)):
            if close.iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]

        signal = direction.copy()
        return pd.DataFrame({"signal": signal}, index=data.index)
