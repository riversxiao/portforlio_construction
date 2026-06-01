"""Pivot Points Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class PivotPoints(Strategy):
    """Pivot point support/resistance strategy.

    Uses classic pivot points (PP, R1, S1) for trading signals.

    Parameters
    ----------
    method : str
        Pivot calculation method (default "classic").
    """

    name = "pivot_points"

    def __init__(self, method: str = "classic", **kwargs):
        super().__init__(method=method, **kwargs)
        self.method = method

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate pivot point signals."""
        high = data["high"].shift(1)
        low = data["low"].shift(1)
        close_prev = data["close"].shift(1)
        close = data["close"]

        pp = (high + low + close_prev) / 3
        r1 = 2 * pp - low
        s1 = 2 * pp - high

        signal = pd.Series(0, index=data.index)
        signal[(close > pp) & (close < r1)] = 1
        signal[(close < pp) & (close > s1)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
