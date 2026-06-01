"""Ultimate Oscillator Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class UltimateOscillator(Strategy):
    """Ultimate Oscillator strategy.

    Combines three timeframes to reduce false signals.

    Parameters
    ----------
    period1 : int
        Short period (default 7).
    period2 : int
        Medium period (default 14).
    period3 : int
        Long period (default 28).
    overbought : float
        Overbought level (default 70).
    oversold : float
        Oversold level (default 30).
    """

    name = "ultimate_oscillator"

    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28,
                 overbought: float = 70, oversold: float = 30, **kwargs):
        super().__init__(period1=period1, period2=period2, period3=period3,
                         overbought=overbought, oversold=oversold, **kwargs)
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Ultimate Oscillator signals."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        avg1 = bp.rolling(self.period1).sum() / (tr.rolling(self.period1).sum() + 1e-10)
        avg2 = bp.rolling(self.period2).sum() / (tr.rolling(self.period2).sum() + 1e-10)
        avg3 = bp.rolling(self.period3).sum() / (tr.rolling(self.period3).sum() + 1e-10)

        uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

        signal = pd.Series(0, index=data.index)
        signal[uo < self.oversold] = 1
        signal[uo > self.overbought] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
