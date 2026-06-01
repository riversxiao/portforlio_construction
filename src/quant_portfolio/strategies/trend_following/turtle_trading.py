"""Turtle Trading Strategy - Donchian channel breakout."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class TurtleTrading(Strategy):
    """Turtle trading strategy using Donchian channel breakouts.

    Buy when price breaks above the N-day high channel,
    sell when price breaks below the N-day low channel.

    Parameters
    ----------
    entry_period : int
        Period for entry channel (default 20).
    exit_period : int
        Period for exit channel (default 10).
    """

    name = "turtle_trading"

    def __init__(self, entry_period: int = 20, exit_period: int = 10, **kwargs):
        super().__init__(entry_period=entry_period, exit_period=exit_period, **kwargs)
        self.entry_period = entry_period
        self.exit_period = exit_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate turtle trading signals."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        upper_channel = high.rolling(self.entry_period).max()
        lower_channel = low.rolling(self.entry_period).min()

        signal = pd.Series(0, index=data.index)
        signal[close >= upper_channel.shift(1)] = 1
        signal[close <= lower_channel.shift(1)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
