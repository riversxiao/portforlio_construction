"""Linear Programming algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from quant_portfolio.algorithms.base import Algorithm


class LinearProgrammingAlgorithm(Algorithm):
    """Linear Programming for Portfolio Constraint Handling.

    Uses LP to maximize expected return subject to linear constraints
    (budget, position limits). Appropriate when the objective is linear
    (risk constraints handled separately):

        max mu^T * w
        s.t. sum(w) = 1, 0 <= w <= max_weight
    """

    name = "linear_programming"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        max_weight = self.params.get("max_weight", 0.5)

        # LP: minimize -mu^T * w (maximize return)
        c = -mu
        # Equality: sum(w) = 1
        A_eq = np.ones((1, n_assets))
        b_eq = np.array([1.0])
        bounds = [(0, max_weight)] * n_assets

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                         method="highs")
        if result.success:
            return result.x
        return np.ones(n_assets) / n_assets
