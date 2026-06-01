"""Dividend Yield Factor Strategy - high dividend yield proxy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class DividendYield(Strategy):
    """Dividend yield factor strategy using price stability proxy.

    Uses stable, low-volatility positive returns as a proxy for
    dividend-paying stocks (income-oriented signal).

    Parameters
    ----------
    window : int
        Window for return and volatility analysis (default 60).
    min_return : float
        Minimum annualized return threshold (default 0.02).
    """

    name = "dividend_yield"

    def __init__(self, window: int = 60, min_return: float = 0.02, **kwargs):
        super().__init__(window=window, min_return=min_return, **kwargs)
        self.window = window
        self.min_return = min_return

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate dividend yield factor signals."""
        close = data["close"]
        returns = close.pct_change()
        avg_ret = returns.rolling(self.window).mean() * 252
        vol = returns.rolling(self.window).std() * np.sqrt(252)

        signal = pd.Series(0, index=data.index)
        signal[(avg_ret > self.min_return) & (vol < 0.2)] = 1
        signal[(avg_ret < -self.min_return) | (vol > 0.5)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
