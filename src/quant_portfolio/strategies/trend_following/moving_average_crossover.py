"""Moving Average Crossover Strategy - dual MA crossover."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    """Dual moving average crossover trend following strategy.

    Buy when fast MA crosses above slow MA, sell when it crosses below.

    Parameters
    ----------
    fast_period : int
        Fast moving average period (default 20).
    slow_period : int
        Slow moving average period (default 50).
    """

    name = "moving_average_crossover"

    def __init__(self, fast_period: int = 20, slow_period: int = 50, **kwargs):
        super().__init__(fast_period=fast_period, slow_period=slow_period, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on MA crossover."""
        close = data["close"]
        fast_ma = close.rolling(self.fast_period).mean()
        slow_ma = close.rolling(self.slow_period).mean()

        signal = pd.Series(0, index=data.index)
        signal[fast_ma > slow_ma] = 1
        signal[fast_ma < slow_ma] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
