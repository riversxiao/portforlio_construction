"""Keltner Channel Strategy - ATR-based channel trend."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class KeltnerChannel(Strategy):
    """Keltner Channel trend following strategy.

    Uses EMA with ATR-based channels for trend signals.

    Parameters
    ----------
    ema_period : int
        EMA period for center line (default 20).
    atr_period : int
        ATR period (default 10).
    multiplier : float
        ATR multiplier for channel width (default 2.0).
    """

    name = "keltner_channel"

    def __init__(self, ema_period: int = 20, atr_period: int = 10,
                 multiplier: float = 2.0, **kwargs):
        super().__init__(ema_period=ema_period, atr_period=atr_period,
                         multiplier=multiplier, **kwargs)
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Keltner Channel signals."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        ema = close.ewm(span=self.ema_period, adjust=False).mean()
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        upper = ema + self.multiplier * atr
        lower = ema - self.multiplier * atr

        signal = pd.Series(0, index=data.index)
        signal[close > upper] = 1
        signal[close < lower] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
