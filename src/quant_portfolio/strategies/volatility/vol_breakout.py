"""Volatility Breakout Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class VolBreakout(Strategy):
    """Volatility breakout strategy.

    Buys when price breaks above previous range scaled by ATR,
    sells when it breaks below.

    Parameters
    ----------
    atr_period : int
        ATR calculation period (default 20).
    multiplier : float
        ATR multiplier for breakout level (default 0.5).
    """

    name = "vol_breakout"

    def __init__(self, atr_period: int = 20, multiplier: float = 0.5, **kwargs):
        super().__init__(atr_period=atr_period, multiplier=multiplier, **kwargs)
        self.atr_period = atr_period
        self.multiplier = multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility breakout signals."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        open_price = data["open"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        upper = open_price + self.multiplier * atr
        lower = open_price - self.multiplier * atr

        signal = pd.Series(0, index=data.index)
        signal[close > upper] = 1
        signal[close < lower] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
