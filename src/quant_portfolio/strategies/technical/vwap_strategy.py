"""VWAP Strategy - Volume Weighted Average Price."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class VWAPStrategy(Strategy):
    """VWAP deviation strategy.

    Trades reversion to VWAP when price deviates significantly.

    Parameters
    ----------
    window : int
        Rolling VWAP window (default 20).
    num_std : float
        Standard deviation threshold (default 2.0).
    """

    name = "vwap_strategy"

    def __init__(self, window: int = 20, num_std: float = 2.0, **kwargs):
        super().__init__(window=window, num_std=num_std, **kwargs)
        self.window = window
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate VWAP deviation signals."""
        tp = (data["high"] + data["low"] + data["close"]) / 3
        volume = data["volume"]

        vwap = (tp * volume).rolling(self.window).sum() / (
            volume.rolling(self.window).sum() + 1e-10
        )
        std = data["close"].rolling(self.window).std()
        upper = vwap + self.num_std * std
        lower = vwap - self.num_std * std

        close = data["close"]
        signal = pd.Series(0, index=data.index)
        signal[close < lower] = 1
        signal[close > upper] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
