"""Momentum Acceleration Strategy - rate of change of momentum."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class MomentumAcceleration(Strategy):
    """Momentum acceleration strategy.

    Measures the second derivative of price (acceleration of momentum).
    Buy when momentum is accelerating positively, sell when decelerating.

    Parameters
    ----------
    lookback : int
        Period for first momentum calculation (default 20).
    accel_period : int
        Period for measuring change in momentum (default 10).
    """

    name = "momentum_acceleration"

    def __init__(self, lookback: int = 20, accel_period: int = 10, **kwargs):
        super().__init__(lookback=lookback, accel_period=accel_period, **kwargs)
        self.lookback = lookback
        self.accel_period = accel_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on momentum acceleration."""
        close = data["close"]
        momentum = close.pct_change(self.lookback)
        acceleration = momentum.diff(self.accel_period)

        signal = pd.Series(0, index=data.index)
        signal[(momentum > 0) & (acceleration > 0)] = 1
        signal[(momentum < 0) & (acceleration < 0)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
