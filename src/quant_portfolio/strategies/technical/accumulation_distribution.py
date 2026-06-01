"""Accumulation/Distribution Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class AccumulationDistribution(Strategy):
    """Accumulation/Distribution line strategy.

    Uses the AD line trend relative to price for divergence signals.

    Parameters
    ----------
    signal_period : int
        Signal smoothing period (default 20).
    """

    name = "accumulation_distribution"

    def __init__(self, signal_period: int = 20, **kwargs):
        super().__init__(signal_period=signal_period, **kwargs)
        self.signal_period = signal_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate A/D line signals."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = mfm * volume
        ad_line = mfv.cumsum()
        ad_ma = ad_line.rolling(self.signal_period).mean()

        signal = pd.Series(0, index=data.index)
        signal[ad_line > ad_ma] = 1
        signal[ad_line < ad_ma] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
