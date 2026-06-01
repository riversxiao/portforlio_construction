"""Gradient Descent Optimizer algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class GradientDescentAlgorithm(Algorithm):
    """Projected Gradient Descent for Portfolio Optimization.

    Minimizes portfolio variance using gradient descent with
    projection onto the simplex (sum to 1, non-negative):

        w_{t+1} = project_simplex(w_t - lr * grad(w_t))

    where grad = 2 * Sigma * w (gradient of w^T * Sigma * w).
    """

    name = "gradient_descent"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values
        lr = self.params.get("learning_rate", 0.01)
        n_iter = self.params.get("n_iter", 500)

        w = np.ones(n_assets) / n_assets

        for _ in range(n_iter):
            grad = 2 * cov @ w
            w = w - lr * grad
            # Project onto simplex
            w = self._project_simplex(w)

        return w

    def _project_simplex(self, v):
        """Project vector onto probability simplex."""
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
        theta = cssv[rho] / (rho + 1.0)
        w = np.maximum(v - theta, 0)
        return w
