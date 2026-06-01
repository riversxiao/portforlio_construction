"""Fibonacci Retracement Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class FibonacciRetracement(Strategy):
    """Fibonacci retracement level strategy.

    Uses key Fibonacci levels from recent high/low for
    support/resistance signals.

    Parameters
    ----------
    lookback : int
        Period to find swing high/low (default 50).
    """

    name = "fibonacci_retracement"

    def __init__(self, lookback: int = 50, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Fibonacci level signals."""
        close = data["close"]
        high = data["high"]
        low = data["low"]

        swing_high = high.rolling(self.lookback).max()
        swing_low = low.rolling(self.lookback).min()
        diff = swing_high - swing_low

        fib_382 = swing_high - 0.382 * diff
        fib_618 = swing_high - 0.618 * diff

        signal = pd.Series(0, index=data.index)
        signal[(close < fib_618) & (close > swing_low)] = 1
        signal[(close > fib_382) & (close < swing_high)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
