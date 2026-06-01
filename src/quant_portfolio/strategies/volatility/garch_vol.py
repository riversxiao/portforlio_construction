"""GARCH Volatility Strategy - volatility prediction."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class GARCHVol(Strategy):
    """GARCH-based volatility prediction strategy.

    Uses a simplified EWMA GARCH(1,1) model to predict volatility
    and trades based on volatility regimes.

    Parameters
    ----------
    omega : float
        GARCH constant (default 0.000001).
    alpha : float
        ARCH parameter (default 0.1).
    beta : float
        GARCH parameter (default 0.85).
    vol_threshold : float
        Volatility threshold multiplier (default 1.5).
    """

    name = "garch_vol"

    def __init__(self, omega: float = 0.000001, alpha: float = 0.1,
                 beta: float = 0.85, vol_threshold: float = 1.5, **kwargs):
        super().__init__(omega=omega, alpha=alpha, beta=beta,
                         vol_threshold=vol_threshold, **kwargs)
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.vol_threshold = vol_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on GARCH volatility forecast."""
        close = data["close"]
        returns = close.pct_change().fillna(0).values
        n = len(returns)

        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns[:20]) if n > 20 else 0.0001

        for i in range(1, n):
            sigma2[i] = (self.omega + self.alpha * returns[i-1]**2 +
                         self.beta * sigma2[i-1])

        vol = np.sqrt(sigma2)
        avg_vol = pd.Series(vol).rolling(60).mean().values
        signal_arr = np.zeros(n)

        for i in range(60, n):
            if avg_vol[i] > 0:
                ratio = vol[i] / avg_vol[i]
                if ratio < 1.0 / self.vol_threshold:
                    signal_arr[i] = 1
                elif ratio > self.vol_threshold:
                    signal_arr[i] = -1

        return pd.DataFrame({"signal": signal_arr}, index=data.index)
