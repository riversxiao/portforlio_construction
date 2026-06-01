"""Adaptive Momentum Strategy - dynamic lookback momentum."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class AdaptiveMomentum(Strategy):
    """Adaptive momentum strategy with dynamic lookback.

    Adjusts the momentum lookback period based on recent volatility.
    Uses shorter lookbacks in high-volatility regimes and longer in low-vol.

    Parameters
    ----------
    min_lookback : int
        Minimum lookback period (default 10).
    max_lookback : int
        Maximum lookback period (default 60).
    vol_window : int
        Window for volatility regime estimation (default 20).
    """

    name = "adaptive_momentum"

    def __init__(self, min_lookback: int = 10, max_lookback: int = 60,
                 vol_window: int = 20, **kwargs):
        super().__init__(min_lookback=min_lookback, max_lookback=max_lookback,
                         vol_window=vol_window, **kwargs)
        self.min_lookback = min_lookback
        self.max_lookback = max_lookback
        self.vol_window = vol_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate adaptive momentum signals with dynamic lookback."""
        close = data["close"]
        returns = close.pct_change()
        vol = returns.rolling(self.vol_window).std()
        vol_pctile = vol.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        signal = pd.Series(0, index=data.index)
        for i in range(self.max_lookback, len(data)):
            pctile = vol_pctile.iloc[i]
            if pd.isna(pctile):
                continue
            lookback = int(self.max_lookback - pctile * (self.max_lookback - self.min_lookback))
            lookback = max(self.min_lookback, min(self.max_lookback, lookback))
            mom = (close.iloc[i] - close.iloc[i - lookback]) / close.iloc[i - lookback]
            if mom > 0.02:
                signal.iloc[i] = 1
            elif mom < -0.02:
                signal.iloc[i] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
