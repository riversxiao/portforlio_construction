"""Smart Beta Strategy - smart beta weighting."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class SmartBeta(Strategy):
    """Smart beta strategy using risk-weighted signals.

    Inverse-volatility weighting concept applied to signal generation.
    Stronger signals in low-volatility environments.

    Parameters
    ----------
    vol_window : int
        Volatility window (default 20).
    signal_window : int
        Signal generation window (default 50).
    """

    name = "smart_beta"

    def __init__(self, vol_window: int = 20, signal_window: int = 50, **kwargs):
        super().__init__(vol_window=vol_window, signal_window=signal_window, **kwargs)
        self.vol_window = vol_window
        self.signal_window = signal_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate smart beta signals."""
        close = data["close"]
        returns = close.pct_change()
        vol = returns.rolling(self.vol_window).std()
        inv_vol = 1.0 / (vol + 1e-10)
        inv_vol_norm = inv_vol / (inv_vol.rolling(252, min_periods=20).mean() + 1e-10)

        trend = np.sign(close - close.rolling(self.signal_window).mean())
        weighted_signal = trend * inv_vol_norm.clip(0, 3)

        signal = pd.Series(0, index=data.index)
        signal[weighted_signal > 1.5] = 1
        signal[weighted_signal < -1.5] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
