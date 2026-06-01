"""Straddle Strategy - synthetic straddle approach."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class StraddleStrategy(Strategy):
    """Synthetic straddle strategy.

    Trades expecting large moves based on vol compression patterns
    (low vol followed by breakout).

    Parameters
    ----------
    vol_window : int
        Vol compression detection window (default 10).
    lookback : int
        Lookback for vol average (default 60).
    compression_threshold : float
        Compression threshold (default 0.5).
    """

    name = "straddle_strategy"

    def __init__(self, vol_window: int = 10, lookback: int = 60,
                 compression_threshold: float = 0.5, **kwargs):
        super().__init__(vol_window=vol_window, lookback=lookback,
                         compression_threshold=compression_threshold, **kwargs)
        self.vol_window = vol_window
        self.lookback = lookback
        self.compression_threshold = compression_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate straddle-like signals."""
        close = data["close"]
        returns = close.pct_change()
        short_vol = returns.rolling(self.vol_window).std()
        long_vol = returns.rolling(self.lookback).std()
        compression = short_vol / (long_vol + 1e-10)

        # After vol compression, trade in direction of breakout
        breakout = close.pct_change(3)

        signal = pd.Series(0, index=data.index)
        compressed = compression < self.compression_threshold
        signal[compressed & (breakout > 0.01)] = 1
        signal[compressed & (breakout < -0.01)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
