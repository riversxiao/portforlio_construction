"""Force Index Strategy - Elder Force Index."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class ForceIndex(Strategy):
    """Elder Force Index strategy.

    Combines price change and volume for force measurement.

    Parameters
    ----------
    period : int
        EMA smoothing period (default 13).
    """

    name = "force_index"

    def __init__(self, period: int = 13, **kwargs):
        super().__init__(period=period, **kwargs)
        self.period = period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Force Index signals."""
        close = data["close"]
        volume = data["volume"]

        fi = close.diff() * volume
        fi_ema = fi.ewm(span=self.period, adjust=False).mean()

        signal = pd.Series(0, index=data.index)
        signal[fi_ema > 0] = 1
        signal[fi_ema < 0] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
