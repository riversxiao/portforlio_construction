"""Idiosyncratic Momentum Strategy - residual momentum after factor adjustment."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class IdiosyncraticMomentum(Strategy):
    """Idiosyncratic (residual) momentum strategy.

    Removes market factor exposure using a rolling beta estimate,
    then trades on the residual momentum.

    Parameters
    ----------
    lookback : int
        Period for residual momentum calculation (default 60).
    beta_window : int
        Rolling window for beta estimation (default 120).
    """

    name = "idiosyncratic_momentum"

    def __init__(self, lookback: int = 60, beta_window: int = 120, **kwargs):
        super().__init__(lookback=lookback, beta_window=beta_window, **kwargs)
        self.lookback = lookback
        self.beta_window = beta_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on residual momentum."""
        close = data["close"]
        returns = close.pct_change()
        market_proxy = returns.rolling(20).mean()

        rolling_cov = returns.rolling(self.beta_window).cov(market_proxy)
        rolling_var = market_proxy.rolling(self.beta_window).var()
        beta = rolling_cov / (rolling_var + 1e-10)

        residual = returns - beta * market_proxy
        residual_momentum = residual.rolling(self.lookback).sum()

        signal = pd.Series(0, index=data.index)
        signal[residual_momentum > 0] = 1
        signal[residual_momentum < 0] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
