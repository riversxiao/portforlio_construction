"""Markowitz Mean-Variance Optimization algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class MeanVarianceOptimization(Algorithm):
    """Markowitz Mean-Variance Optimization.

    Finds the portfolio on the efficient frontier that maximizes expected
    return for a given level of risk, or equivalently minimizes risk for
    a given level of return. The objective is:

        min w^T * Sigma * w - risk_aversion * mu^T * w

    subject to sum(w) = 1 and w >= 0 (if long_only).
    """

    name = "mean_variance"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        risk_aversion = self.params.get("risk_aversion", 1.0)

        def objective(w):
            return 0.5 * risk_aversion * w @ cov @ w - mu @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
