"""Volatility Regime Strategy - regime switching."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class VolRegime(Strategy):
    """Volatility regime switching strategy.

    Identifies low/high volatility regimes and trades accordingly.
    Long in low-vol regimes, defensive in high-vol.

    Parameters
    ----------
    vol_window : int
        Volatility estimation window (default 20).
    regime_window : int
        Regime classification window (default 120).
    """

    name = "vol_regime"

    def __init__(self, vol_window: int = 20, regime_window: int = 120, **kwargs):
        super().__init__(vol_window=vol_window, regime_window=regime_window, **kwargs)
        self.vol_window = vol_window
        self.regime_window = regime_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate vol regime signals."""
        close = data["close"]
        returns = close.pct_change()
        vol = returns.rolling(self.vol_window).std()
        vol_median = vol.rolling(self.regime_window).median()

        trend = np.sign(close - close.rolling(20).mean())

        signal = pd.Series(0, index=data.index)
        low_vol = vol < vol_median
        high_vol = vol > vol_median * 1.5

        signal[low_vol & (trend > 0)] = 1
        signal[high_vol & (trend < 0)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
