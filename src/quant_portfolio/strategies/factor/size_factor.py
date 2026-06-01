"""Size Factor Strategy - small-cap premium proxy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class SizeFactor(Strategy):
    """Size factor strategy using volume as market cap proxy.

    Uses relative volume as a proxy for market capitalization.
    Lower relative volume indicates smaller companies.

    Parameters
    ----------
    vol_window : int
        Volume averaging window (default 60).
    threshold : float
        Relative volume threshold (default 0.5).
    """

    name = "size_factor"

    def __init__(self, vol_window: int = 60, threshold: float = 0.5, **kwargs):
        super().__init__(vol_window=vol_window, threshold=threshold, **kwargs)
        self.vol_window = vol_window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate size factor signals."""
        close = data["close"]
        volume = data["volume"]
        avg_vol = volume.rolling(self.vol_window).mean()
        vol_rank = avg_vol.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        returns_20d = close.pct_change(20)

        signal = pd.Series(0, index=data.index)
        signal[(vol_rank < self.threshold) & (returns_20d > 0)] = 1
        signal[(vol_rank > 1 - self.threshold) & (returns_20d < 0)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
