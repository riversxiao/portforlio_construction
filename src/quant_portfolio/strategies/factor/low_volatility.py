"""Low Volatility Factor Strategy - low-vol anomaly."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class LowVolatility(Strategy):
    """Low volatility anomaly strategy.

    Buys during low-volatility periods (anomaly shows low-vol
    assets outperform on risk-adjusted basis).

    Parameters
    ----------
    vol_window : int
        Volatility measurement window (default 60).
    threshold_pct : float
        Percentile threshold for low vol (default 0.3).
    """

    name = "low_volatility"

    def __init__(self, vol_window: int = 60, threshold_pct: float = 0.3, **kwargs):
        super().__init__(vol_window=vol_window, threshold_pct=threshold_pct, **kwargs)
        self.vol_window = vol_window
        self.threshold_pct = threshold_pct

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate low volatility factor signals."""
        close = data["close"]
        returns = close.pct_change()
        vol = returns.rolling(self.vol_window).std()
        vol_rank = vol.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        signal = pd.Series(0, index=data.index)
        signal[vol_rank < self.threshold_pct] = 1
        signal[vol_rank > 1 - self.threshold_pct] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
