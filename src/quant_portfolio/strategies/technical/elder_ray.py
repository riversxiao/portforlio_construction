"""Elder Ray Strategy - bull/bear power."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class ElderRay(Strategy):
    """Elder Ray bull/bear power strategy.

    Uses bull power (high - EMA) and bear power (low - EMA) to gauge
    the strength of bulls and bears.

    Parameters
    ----------
    ema_period : int
        EMA period (default 13).
    """

    name = "elder_ray"

    def __init__(self, ema_period: int = 13, **kwargs):
        super().__init__(ema_period=ema_period, **kwargs)
        self.ema_period = ema_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Elder Ray signals."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        ema = close.ewm(span=self.ema_period, adjust=False).mean()
        bull_power = high - ema
        bear_power = low - ema

        signal = pd.Series(0, index=data.index)
        signal[(bull_power > 0) & (bear_power > -0.01 * close)] = 1
        signal[(bear_power < 0) & (bull_power < 0.01 * close)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
