"""Robust Mean-Variance Optimization with uncertainty sets."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class RobustOptimization(Algorithm):
    """Robust Mean-Variance Optimization.

    Accounts for estimation uncertainty in expected returns by using an
    ellipsoidal uncertainty set. The worst-case return within the set is:

        min_{mu in U} mu^T * w = hat_mu^T * w - kappa * ||Sigma^{1/2} * w||

    where kappa controls the size of the uncertainty set. This leads to a
    more conservative allocation that is robust to estimation errors.
    """

    name = "robust_optimization"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        kappa = self.params.get("kappa", 0.5)
        risk_aversion = self.params.get("risk_aversion", 1.0)

        def objective(w):
            port_var = w @ cov @ w
            port_vol = np.sqrt(port_var) if port_var > 0 else 0
            # Worst-case expected return
            worst_ret = mu @ w - kappa * port_vol
            return -worst_ret + 0.5 * risk_aversion * port_var

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
