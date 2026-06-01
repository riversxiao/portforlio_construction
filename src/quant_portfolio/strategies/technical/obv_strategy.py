"""On-Balance Volume Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class OBVStrategy(Strategy):
    """On-Balance Volume (OBV) strategy.

    Uses OBV trend to confirm price direction.

    Parameters
    ----------
    signal_period : int
        Period for OBV signal line (default 20).
    """

    name = "obv_strategy"

    def __init__(self, signal_period: int = 20, **kwargs):
        super().__init__(signal_period=signal_period, **kwargs)
        self.signal_period = signal_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate OBV-based signals."""
        close = data["close"]
        volume = data["volume"]

        direction = np.sign(close.diff())
        obv = (volume * direction).cumsum()
        obv_ma = obv.rolling(self.signal_period).mean()

        signal = pd.Series(0, index=data.index)
        signal[obv > obv_ma] = 1
        signal[obv < obv_ma] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
