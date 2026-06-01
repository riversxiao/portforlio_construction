"""Factor Risk Parity algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class FactorRiskParityOptimization(Algorithm):
    """Factor Risk Parity.

    Extends risk parity to factor space. Decomposes the covariance
    matrix using PCA into factor contributions, then equalizes risk
    contribution across principal factors rather than individual assets.

    Steps:
    1. PCA decomposition of covariance: Sigma = V * Lambda * V^T
    2. Transform weights to factor space: w_f = V^T * w
    3. Equalize risk contribution in factor space
    """

    name = "factor_risk_parity"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values
        n_factors = min(self.params.get("n_factors", 3), n_assets)

        # PCA decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx][:n_factors]
        eigenvectors = eigenvectors[:, idx][:, :n_factors]

        def objective(w):
            # Factor exposures
            factor_exp = eigenvectors.T @ w
            # Factor risk contributions
            factor_var = eigenvalues * factor_exp ** 2
            total_var = factor_var.sum()
            if total_var < 1e-10:
                return 0.0
            target = total_var / n_factors
            return np.sum((factor_var - target) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
