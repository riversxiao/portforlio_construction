"""Target Return Optimization algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class TargetReturnOptimization(Algorithm):
    """Target Return Portfolio Optimization.

    Finds the minimum variance portfolio subject to achieving a
    specified target return:

        min w^T * Sigma * w
        s.t. mu^T * w >= target_return
             sum(w) = 1, w >= 0
    """

    name = "target_return"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        target = self.params.get("target_return", mu.mean())

        def objective(w):
            return w @ cov @ w

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: mu @ w - target},
        ]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
