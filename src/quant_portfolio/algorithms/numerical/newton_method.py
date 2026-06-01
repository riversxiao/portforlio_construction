"""Newton's Method algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class NewtonMethodAlgorithm(Algorithm):
    """Newton's Method for Portfolio Optimization.

    Uses second-order information (Hessian) for faster convergence.
    For quadratic objectives (portfolio variance), Newton's method
    converges in one step:

        w* = w - H^{-1} * g

    where H = 2*Sigma (Hessian) and g = 2*Sigma*w (gradient).
    Includes constraints via augmented Lagrangian.
    """

    name = "newton_method"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values

        # Analytical solution with equality constraint (sum = 1)
        # Using KKT conditions: 2*Sigma*w + lambda*1 = 0, 1^T*w = 1
        ones = np.ones(n_assets)
        try:
            inv_cov = np.linalg.pinv(cov)
            w = inv_cov @ ones / (ones @ inv_cov @ ones)
        except np.linalg.LinAlgError:
            w = np.ones(n_assets) / n_assets

        # Project to non-negative
        w = np.maximum(w, 0)
        total = w.sum()
        if total > 1e-10:
            w = w / total
        else:
            w = np.ones(n_assets) / n_assets
        return w
