"""Minimum Correlation Allocation algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class MinCorrelationAlgorithm(Algorithm):
    """Minimum Correlation Portfolio.

    Finds weights that minimize the weighted average pairwise
    correlation in the portfolio. This maximizes diversification
    by seeking assets that move independently:

        min sum_i sum_j w_i * w_j * rho_ij
    """

    name = "min_correlation"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        corr = returns.corr().values

        def objective(w):
            return w @ corr @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
