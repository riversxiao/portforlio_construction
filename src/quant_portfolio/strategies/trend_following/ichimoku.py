"""Ichimoku Cloud Strategy - Ichimoku Kinko Hyo."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class Ichimoku(Strategy):
    """Ichimoku Cloud trading strategy.

    Uses the Tenkan-sen/Kijun-sen crossover and cloud position
    for trend signals.

    Parameters
    ----------
    tenkan_period : int
        Tenkan-sen (conversion line) period (default 9).
    kijun_period : int
        Kijun-sen (base line) period (default 26).
    senkou_b_period : int
        Senkou Span B period (default 52).
    """

    name = "ichimoku"

    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26,
                 senkou_b_period: int = 52, **kwargs):
        super().__init__(tenkan_period=tenkan_period, kijun_period=kijun_period,
                         senkou_b_period=senkou_b_period, **kwargs)
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Ichimoku cloud signals."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tenkan = (high.rolling(self.tenkan_period).max() +
                  low.rolling(self.tenkan_period).min()) / 2
        kijun = (high.rolling(self.kijun_period).max() +
                 low.rolling(self.kijun_period).min()) / 2
        senkou_a = (tenkan + kijun) / 2
        senkou_b = (high.rolling(self.senkou_b_period).max() +
                    low.rolling(self.senkou_b_period).min()) / 2

        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

        signal = pd.Series(0, index=data.index)
        signal[(tenkan > kijun) & (close > cloud_top)] = 1
        signal[(tenkan < kijun) & (close < cloud_bottom)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
