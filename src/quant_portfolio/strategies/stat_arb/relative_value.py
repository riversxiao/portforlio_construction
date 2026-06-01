"""Relative Value Strategy - relative value convergence."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class RelativeValue(Strategy):
    """Relative value convergence strategy.

    Trades when the ratio of price to its exponential moving average
    deviates beyond thresholds.

    Parameters
    ----------
    ema_span : int
        EMA span for fair value estimate (default 50).
    entry_z : float
        Z-score entry threshold (default 2.0).
    lookback : int
        Window for z-score calculation (default 30).
    """

    name = "relative_value"

    def __init__(self, ema_span: int = 50, entry_z: float = 2.0,
                 lookback: int = 30, **kwargs):
        super().__init__(ema_span=ema_span, entry_z=entry_z, lookback=lookback, **kwargs)
        self.ema_span = ema_span
        self.entry_z = entry_z
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate relative value signals."""
        close = data["close"]
        ema = close.ewm(span=self.ema_span, adjust=False).mean()
        ratio = close / (ema + 1e-10)
        ratio_mean = ratio.rolling(self.lookback).mean()
        ratio_std = ratio.rolling(self.lookback).std()
        zscore = (ratio - ratio_mean) / (ratio_std + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[zscore < -self.entry_z] = 1
        signal[zscore > self.entry_z] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
