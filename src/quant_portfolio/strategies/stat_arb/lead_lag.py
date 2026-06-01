"""Lead-Lag Strategy - lead-lag relationship exploitation."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class LeadLag(Strategy):
    """Lead-lag relationship exploitation strategy.

    Uses lagged autocorrelation to detect predictability in returns
    and trades based on the predictive signal.

    Parameters
    ----------
    lag : int
        Lag period for autocorrelation (default 1).
    window : int
        Rolling window for signal estimation (default 20).
    threshold : float
        Signal threshold (default 0.0).
    """

    name = "lead_lag"

    def __init__(self, lag: int = 1, window: int = 20,
                 threshold: float = 0.0, **kwargs):
        super().__init__(lag=lag, window=window, threshold=threshold, **kwargs)
        self.lag = lag
        self.window = window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate lead-lag signals."""
        close = data["close"]
        returns = close.pct_change()
        lagged_returns = returns.shift(self.lag)
        rolling_corr = returns.rolling(self.window).corr(lagged_returns)

        predictor = lagged_returns * rolling_corr

        signal = pd.Series(0, index=data.index)
        signal[predictor > self.threshold] = 1
        signal[predictor < -self.threshold] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
