"""Covariance Estimation (Ledoit-Wolf Shrinkage) algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class CovarianceEstimationAlgorithm(Algorithm):
    """Ledoit-Wolf Shrinkage Covariance Estimation.

    Applies shrinkage to the sample covariance matrix to reduce
    estimation error:

        Sigma_shrunk = (1-delta) * S + delta * F

    where S is the sample covariance, F is the shrinkage target
    (scaled identity), and delta is the optimal shrinkage intensity.
    Then uses minimum variance with the improved estimate.
    """

    name = "covariance_estimation"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_obs = len(ret_matrix)

        # Sample covariance
        S = np.cov(ret_matrix, rowvar=False)
        # Shrinkage target: scaled identity
        mu_target = np.trace(S) / n_assets
        F = mu_target * np.eye(n_assets)

        # Ledoit-Wolf optimal shrinkage intensity
        X = ret_matrix - ret_matrix.mean(axis=0)
        S2 = (X.T @ X) / n_obs
        # Sum of squared off-diagonal elements of sample cov
        d2 = np.sum((S2 - F) ** 2) / n_assets
        # Estimate of b2
        b_bar = 0.0
        for k in range(n_obs):
            xk = X[k:k+1]
            mk = (xk.T @ xk) - S2
            b_bar += np.sum(mk ** 2) / n_assets
        b_bar /= n_obs ** 2

        delta = min(max(b_bar / d2, 0), 1) if d2 > 1e-10 else 0.5

        cov_shrunk = (1 - delta) * S + delta * F

        # Minimum variance with shrunk covariance
        def objective(w):
            return w @ cov_shrunk @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
