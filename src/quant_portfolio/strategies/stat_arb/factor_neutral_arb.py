"""Factor-Neutral Arbitrage Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class FactorNeutralArb(Strategy):
    """Factor-neutral statistical arbitrage strategy.

    Hedges market factor exposure and trades the residual alpha.

    Parameters
    ----------
    window : int
        Rolling window for beta estimation (default 60).
    entry_z : float
        Z-score threshold for residual entry (default 1.5).
    """

    name = "factor_neutral_arb"

    def __init__(self, window: int = 60, entry_z: float = 1.5, **kwargs):
        super().__init__(window=window, entry_z=entry_z, **kwargs)
        self.window = window
        self.entry_z = entry_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate factor-neutral arb signals."""
        close = data["close"]
        returns = close.pct_change()
        market = returns.rolling(20).mean()  # Market proxy

        rolling_cov = returns.rolling(self.window).cov(market)
        rolling_var = market.rolling(self.window).var()
        beta = rolling_cov / (rolling_var + 1e-10)
        residual = returns - beta * market

        residual_cum = residual.rolling(self.window).sum()
        res_std = residual_cum.rolling(self.window).std()
        zscore = residual_cum / (res_std + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[zscore < -self.entry_z] = 1
        signal[zscore > self.entry_z] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
