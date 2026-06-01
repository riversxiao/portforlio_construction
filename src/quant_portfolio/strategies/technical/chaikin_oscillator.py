"""Chaikin Oscillator Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class ChaikinOscillator(Strategy):
    """Chaikin Oscillator strategy.

    Measures the MACD of the Accumulation/Distribution line.

    Parameters
    ----------
    fast_period : int
        Fast EMA period (default 3).
    slow_period : int
        Slow EMA period (default 10).
    """

    name = "chaikin_oscillator"

    def __init__(self, fast_period: int = 3, slow_period: int = 10, **kwargs):
        super().__init__(fast_period=fast_period, slow_period=slow_period, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Chaikin Oscillator signals."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = mfm * volume
        ad_line = mfv.cumsum()

        fast_ema = ad_line.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = ad_line.ewm(span=self.slow_period, adjust=False).mean()
        chaikin = fast_ema - slow_ema

        signal = pd.Series(0, index=data.index)
        signal[chaikin > 0] = 1
        signal[chaikin < 0] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
