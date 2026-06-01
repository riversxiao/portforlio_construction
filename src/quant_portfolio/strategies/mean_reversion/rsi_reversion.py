"""RSI Reversion Strategy - oversold/overbought mean reversion."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class RSIReversion(Strategy):
    """RSI-based mean reversion strategy.

    Buys when RSI indicates oversold conditions and sells when overbought,
    expecting mean reversion.

    Parameters
    ----------
    period : int
        RSI calculation period (default 14).
    oversold : float
        Oversold threshold (default 30).
    overbought : float
        Overbought threshold (default 70).
    """

    name = "rsi_reversion"

    def __init__(self, period: int = 14, oversold: float = 30,
                 overbought: float = 70, **kwargs):
        super().__init__(period=period, oversold=oversold, overbought=overbought, **kwargs)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI reversion signals."""
        close = data["close"]
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.rolling(self.period).mean()
        avg_loss = loss.rolling(self.period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        signal = pd.Series(0, index=data.index)
        signal[rsi < self.oversold] = 1
        signal[rsi > self.overbought] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
