"""Global Minimum Variance Portfolio algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class MinVarianceOptimization(Algorithm):
    """Global Minimum Variance Portfolio.

    Finds the portfolio with the lowest possible variance regardless of
    expected return. The objective is:

        min w^T * Sigma * w
        s.t. sum(w) = 1, w >= 0
    """

    name = "min_variance"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values

        def objective(w):
            return w @ cov @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
