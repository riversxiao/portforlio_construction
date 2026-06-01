"""Pairs Trading Strategy - cointegrated pairs mean reversion."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class PairsTrading(Strategy):
    """Pairs trading strategy using spread z-score.

    For single-asset input, uses a synthetic pair created from the price
    and its moving average. Trades mean reversion of the spread.

    Parameters
    ----------
    window : int
        Rolling window for spread statistics (default 60).
    entry_z : float
        Z-score threshold for entry (default 2.0).
    exit_z : float
        Z-score threshold for exit (default 0.5).
    """

    name = "pairs_trading"

    def __init__(self, window: int = 60, entry_z: float = 2.0,
                 exit_z: float = 0.5, **kwargs):
        super().__init__(window=window, entry_z=entry_z, exit_z=exit_z, **kwargs)
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate pairs trading signals using spread z-score."""
        close = data["close"]
        ma = close.rolling(self.window).mean()
        spread = close - ma
        spread_mean = spread.rolling(self.window).mean()
        spread_std = spread.rolling(self.window).std()
        zscore = (spread - spread_mean) / (spread_std + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[zscore < -self.entry_z] = 1
        signal[zscore > self.entry_z] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
