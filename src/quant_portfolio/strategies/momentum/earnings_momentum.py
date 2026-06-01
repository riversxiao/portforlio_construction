"""Earnings Momentum Strategy - momentum based on earnings surprise proxy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class EarningsMomentum(Strategy):
    """Earnings momentum strategy using volume-price proxy for surprise.

    Uses abnormal volume combined with price change as a proxy for
    earnings surprise momentum when fundamental data is not available.

    Parameters
    ----------
    lookback : int
        Period for volume average calculation (default 20).
    price_period : int
        Period for price momentum measurement (default 5).
    vol_threshold : float
        Volume multiplier threshold for abnormal volume (default 1.5).
    """

    name = "earnings_momentum"

    def __init__(self, lookback: int = 20, price_period: int = 5,
                 vol_threshold: float = 1.5, **kwargs):
        super().__init__(lookback=lookback, price_period=price_period,
                         vol_threshold=vol_threshold, **kwargs)
        self.lookback = lookback
        self.price_period = price_period
        self.vol_threshold = vol_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on earnings momentum proxy."""
        close = data["close"]
        volume = data["volume"]

        avg_volume = volume.rolling(self.lookback).mean()
        vol_ratio = volume / (avg_volume + 1e-10)
        price_change = close.pct_change(self.price_period)

        abnormal_vol = vol_ratio > self.vol_threshold

        signal = pd.Series(0, index=data.index)
        signal[abnormal_vol & (price_change > 0)] = 1
        signal[abnormal_vol & (price_change < 0)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
