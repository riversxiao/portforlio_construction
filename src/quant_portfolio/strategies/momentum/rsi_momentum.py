"""RSI Momentum Strategy - momentum signals using RSI indicator."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class RSIMomentum(Strategy):
    """RSI-based momentum strategy.

    Generates buy signals when RSI crosses above a mid-level threshold
    from below (momentum confirmation) and sell when it crosses below.

    Parameters
    ----------
    period : int
        RSI calculation period (default 14).
    upper : float
        Upper threshold for strong momentum (default 60).
    lower : float
        Lower threshold for weak momentum (default 40).
    """

    name = "rsi_momentum"

    def __init__(self, period: int = 14, upper: float = 60, lower: float = 40, **kwargs):
        super().__init__(period=period, upper=upper, lower=lower, **kwargs)
        self.period = period
        self.upper = upper
        self.lower = lower

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on RSI momentum."""
        close = data["close"]
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.rolling(self.period).mean()
        avg_loss = loss.rolling(self.period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        signal = pd.Series(0, index=data.index)
        signal[rsi > self.upper] = 1
        signal[rsi < self.lower] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
