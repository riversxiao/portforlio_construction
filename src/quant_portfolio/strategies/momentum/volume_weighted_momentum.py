"""Volume Weighted Momentum Strategy - momentum confirmed by volume."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class VolumeWeightedMomentum(Strategy):
    """Volume-weighted momentum strategy.

    Weights price momentum by relative volume to confirm signals.
    Higher volume gives more weight to the momentum signal.

    Parameters
    ----------
    lookback : int
        Momentum lookback period (default 20).
    vol_lookback : int
        Volume average lookback (default 20).
    """

    name = "volume_weighted_momentum"

    def __init__(self, lookback: int = 20, vol_lookback: int = 20, **kwargs):
        super().__init__(lookback=lookback, vol_lookback=vol_lookback, **kwargs)
        self.lookback = lookback
        self.vol_lookback = vol_lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-weighted momentum signals."""
        close = data["close"]
        volume = data["volume"]

        momentum = close.pct_change(self.lookback)
        avg_vol = volume.rolling(self.vol_lookback).mean()
        vol_ratio = volume / (avg_vol + 1e-10)

        vw_momentum = momentum * vol_ratio

        signal = pd.Series(0, index=data.index)
        signal[vw_momentum > 0.02] = 1
        signal[vw_momentum < -0.02] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
