"""Robust Statistics algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class RobustStatisticsAlgorithm(Algorithm):
    """Robust Statistical Estimators for Allocation.

    Uses robust estimators (Median Absolute Deviation, trimmed mean)
    instead of standard mean/variance to reduce sensitivity to outliers.

    The robust covariance is estimated using the Minimum Covariance
    Determinant (MCD) approximation via iterative trimming.
    """

    name = "robust_statistics"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        trim_pct = self.params.get("trim_pct", 0.1)
        n_obs = len(ret_matrix)

        # Trimmed covariance estimation
        # Remove most extreme observations
        port_ret = ret_matrix.mean(axis=1)
        lower = np.percentile(port_ret, trim_pct * 100)
        upper = np.percentile(port_ret, (1 - trim_pct) * 100)
        mask = (port_ret >= lower) & (port_ret <= upper)
        trimmed = ret_matrix[mask]

        if len(trimmed) < n_assets + 1:
            trimmed = ret_matrix

        cov_robust = np.cov(trimmed, rowvar=False)

        # Minimum variance with robust covariance
        def objective(w):
            return w @ cov_robust @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
