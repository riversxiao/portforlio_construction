"""Maximum Diversification Portfolio algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class MaxDiversificationOptimization(Algorithm):
    """Maximum Diversification Portfolio.

    Maximizes the diversification ratio:

        DR = (w^T * sigma) / sqrt(w^T * Sigma * w)

    where sigma is the vector of individual asset volatilities and Sigma
    is the covariance matrix. A higher DR indicates more diversification.
    """

    name = "max_diversification"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values
        vols = np.sqrt(np.diag(cov))

        def neg_div_ratio(w):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 0.0
            return -(w @ vols) / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(neg_div_ratio, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
