"""Maximum Sharpe Ratio Portfolio algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class MaxSharpeOptimization(Algorithm):
    """Maximum Sharpe Ratio Portfolio.

    Maximizes the Sharpe ratio (excess return per unit risk):

        max (mu^T * w - rf) / sqrt(w^T * Sigma * w)

    Solved via the equivalent formulation of minimizing negative Sharpe.
    """

    name = "max_sharpe"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        rf = self.params.get("risk_free_rate", 0.0)

        def neg_sharpe(w):
            port_ret = mu @ w - rf
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 0.0
            return -port_ret / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(neg_sharpe, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
