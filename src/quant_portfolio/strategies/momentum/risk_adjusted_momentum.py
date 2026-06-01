"""Risk-Adjusted Momentum Strategy - Sharpe ratio based momentum ranking."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class RiskAdjustedMomentum(Strategy):
    """Risk-adjusted momentum strategy.

    Uses the rolling Sharpe ratio (return/volatility) as the momentum
    signal instead of raw returns.

    Parameters
    ----------
    lookback : int
        Period for Sharpe ratio calculation (default 60).
    threshold : float
        Sharpe threshold for signal generation (default 0.5).
    """

    name = "risk_adjusted_momentum"

    def __init__(self, lookback: int = 60, threshold: float = 0.5, **kwargs):
        super().__init__(lookback=lookback, threshold=threshold, **kwargs)
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on rolling Sharpe ratio."""
        close = data["close"]
        returns = close.pct_change()

        rolling_mean = returns.rolling(self.lookback).mean()
        rolling_std = returns.rolling(self.lookback).std()
        sharpe = (rolling_mean / (rolling_std + 1e-10)) * np.sqrt(252)

        signal = pd.Series(0, index=data.index)
        signal[sharpe > self.threshold] = 1
        signal[sharpe < -self.threshold] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
