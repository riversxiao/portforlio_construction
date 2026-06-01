"""Volatility Targeting Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class VolTargeting(Strategy):
    """Volatility targeting strategy.

    Adjusts position based on realized volatility relative to target.
    Reduces exposure in high-vol and increases in low-vol.

    Parameters
    ----------
    target_vol : float
        Target annualized volatility (default 0.15).
    vol_window : int
        Volatility estimation window (default 20).
    """

    name = "vol_targeting"

    def __init__(self, target_vol: float = 0.15, vol_window: int = 20, **kwargs):
        super().__init__(target_vol=target_vol, vol_window=vol_window, **kwargs)
        self.target_vol = target_vol
        self.vol_window = vol_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate vol-targeted signals."""
        close = data["close"]
        returns = close.pct_change()
        realized_vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        trend = np.sign(close - close.rolling(50).mean())

        # Scale signal by vol ratio
        vol_ratio = self.target_vol / (realized_vol + 1e-10)
        vol_ratio = vol_ratio.clip(0.2, 3.0)

        raw_signal = trend * vol_ratio
        signal = pd.Series(0, index=data.index)
        signal[raw_signal > 1.0] = 1
        signal[raw_signal < -1.0] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
