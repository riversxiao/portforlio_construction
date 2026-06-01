"""MACD Momentum Strategy - momentum signals using MACD crossover."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class MACDMomentum(Strategy):
    """MACD crossover momentum strategy.

    Generates buy signals when MACD line crosses above signal line
    and sell signals when it crosses below.

    Parameters
    ----------
    fast_period : int
        Fast EMA period (default 12).
    slow_period : int
        Slow EMA period (default 26).
    signal_period : int
        Signal line EMA period (default 9).
    """

    name = "macd_momentum"

    def __init__(self, fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9, **kwargs):
        super().__init__(fast_period=fast_period, slow_period=slow_period,
                         signal_period=signal_period, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on MACD crossover."""
        close = data["close"]

        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        macd_diff = macd_line - signal_line

        signal = pd.Series(0, index=data.index)
        signal[macd_diff > 0] = 1
        signal[macd_diff < 0] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
