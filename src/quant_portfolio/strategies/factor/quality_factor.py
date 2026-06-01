"""Quality Factor Strategy - profitability/quality proxy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class QualityFactor(Strategy):
    """Quality factor strategy using price stability as quality proxy.

    Uses low volatility and consistent positive returns as indicators
    of underlying business quality.

    Parameters
    ----------
    vol_window : int
        Volatility measurement window (default 60).
    return_window : int
        Return consistency window (default 252).
    """

    name = "quality_factor"

    def __init__(self, vol_window: int = 60, return_window: int = 252, **kwargs):
        super().__init__(vol_window=vol_window, return_window=return_window, **kwargs)
        self.vol_window = vol_window
        self.return_window = return_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate quality factor signals."""
        close = data["close"]
        returns = close.pct_change()
        vol = returns.rolling(self.vol_window).std()
        avg_return = returns.rolling(self.return_window).mean()
        sharpe = avg_return / (vol + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[sharpe > 0.05] = 1
        signal[sharpe < -0.05] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
