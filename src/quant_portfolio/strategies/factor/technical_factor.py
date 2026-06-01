"""Technical Factor Strategy - composite technical score."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class TechnicalFactor(Strategy):
    """Composite technical factor strategy.

    Combines multiple technical indicators into a single factor score.

    Parameters
    ----------
    rsi_period : int
        RSI period (default 14).
    ma_period : int
        Moving average period (default 50).
    """

    name = "technical_factor"

    def __init__(self, rsi_period: int = 14, ma_period: int = 50, **kwargs):
        super().__init__(rsi_period=rsi_period, ma_period=ma_period, **kwargs)
        self.rsi_period = rsi_period
        self.ma_period = ma_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate composite technical factor signals."""
        close = data["close"]
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        ma = close.rolling(self.ma_period).mean()
        ma_signal = np.sign(close - ma)

        rsi_score = (rsi - 50) / 50  # -1 to 1

        composite = (ma_signal + rsi_score) / 2
        signal = pd.Series(0, index=data.index)
        signal[composite > 0.3] = 1
        signal[composite < -0.3] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
