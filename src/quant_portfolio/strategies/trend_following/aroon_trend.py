"""Aroon Trend Strategy - Aroon indicator trend."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class AroonTrend(Strategy):
    """Aroon indicator trend strategy.

    Uses Aroon Up and Aroon Down to determine trend direction.

    Parameters
    ----------
    period : int
        Aroon calculation period (default 25).
    threshold : float
        Aroon threshold for strong trend (default 70).
    """

    name = "aroon_trend"

    def __init__(self, period: int = 25, threshold: float = 70, **kwargs):
        super().__init__(period=period, threshold=threshold, **kwargs)
        self.period = period
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Aroon trend signals."""
        high = data["high"]
        low = data["low"]

        aroon_up = high.rolling(self.period + 1).apply(
            lambda x: x.argmax() / self.period * 100, raw=True
        )
        aroon_down = low.rolling(self.period + 1).apply(
            lambda x: x.argmin() / self.period * 100, raw=True
        )

        signal = pd.Series(0, index=data.index)
        signal[(aroon_up > self.threshold) & (aroon_down < 100 - self.threshold)] = 1
        signal[(aroon_down > self.threshold) & (aroon_up < 100 - self.threshold)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
