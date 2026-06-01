"""Quadratic Programming algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class QuadraticProgrammingAlgorithm(Algorithm):
    """Quadratic Programming (QP) Solver for Portfolios.

    Solves the standard QP formulation of portfolio optimization:

        min  (1/2) * w^T * Q * w + c^T * w
        s.t. A_eq * w = b_eq
             w >= 0

    where Q = covariance matrix, c = -mu (negative expected returns),
    and constraints enforce full investment.
    """

    name = "quadratic_programming"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        risk_aversion = self.params.get("risk_aversion", 1.0)

        def objective(w):
            return 0.5 * risk_aversion * w @ cov @ w - mu @ w

        def grad(w):
            return risk_aversion * cov @ w - mu

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                        "jac": lambda w: np.ones(n_assets)}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP", jac=grad,
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
