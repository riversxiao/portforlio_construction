"""Dispersion Trading Strategy - index vs components dispersion."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class DispersionTrading(Strategy):
    """Dispersion trading strategy.

    Trades based on the relationship between realized volatility
    and its moving average (dispersion proxy).

    Parameters
    ----------
    vol_window : int
        Window for realized volatility (default 20).
    ma_window : int
        Window for vol moving average (default 60).
    threshold : float
        Dispersion threshold multiplier (default 1.5).
    """

    name = "dispersion_trading"

    def __init__(self, vol_window: int = 20, ma_window: int = 60,
                 threshold: float = 1.5, **kwargs):
        super().__init__(vol_window=vol_window, ma_window=ma_window,
                         threshold=threshold, **kwargs)
        self.vol_window = vol_window
        self.ma_window = ma_window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate dispersion trading signals."""
        close = data["close"]
        returns = close.pct_change()
        realized_vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        avg_vol = realized_vol.rolling(self.ma_window).mean()
        ratio = realized_vol / (avg_vol + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[ratio > self.threshold] = -1
        signal[ratio < 1.0 / self.threshold] = 1

        return pd.DataFrame({"signal": signal}, index=data.index)
