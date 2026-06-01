"""Convex Optimization Wrapper algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class ConvexOptimizationAlgorithm(Algorithm):
    """General Convex Optimization Wrapper.

    Solves convex portfolio optimization problems with multiple
    objectives (return, variance, CVaR) combined via scalarization:

        min lambda_1 * variance + lambda_2 * (-return) + lambda_3 * CVaR

    Uses scipy's trust-constr method for convex problems.
    """

    name = "convex_optimization"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        ret_matrix = returns.values
        lambda_var = self.params.get("lambda_var", 1.0)
        lambda_ret = self.params.get("lambda_ret", 1.0)
        lambda_cvar = self.params.get("lambda_cvar", 0.5)
        alpha = self.params.get("alpha", 0.05)

        def objective(w):
            variance = w @ cov @ w
            neg_return = -mu @ w
            # CVaR
            port_ret = ret_matrix @ w
            sorted_ret = np.sort(port_ret)
            n_tail = max(1, int(len(sorted_ret) * alpha))
            cvar = -sorted_ret[:n_tail].mean()

            return (lambda_var * variance + lambda_ret * neg_return +
                    lambda_cvar * cvar)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
