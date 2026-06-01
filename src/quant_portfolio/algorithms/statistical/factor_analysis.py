"""Factor Analysis algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class FactorAnalysisAlgorithm(Algorithm):
    """Statistical Factor Analysis for Allocation.

    Performs factor analysis to decompose the covariance into
    common factors and specific variances:

        Sigma = L*L^T + Psi

    where L are factor loadings and Psi is the diagonal uniqueness
    matrix. Allocates to maximize diversification across factors.
    """

    name = "factor_analysis"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values
        n_factors = min(self.params.get("n_factors", 3), n_assets - 1)

        # Extract factors via eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Factor loadings
        L = eigenvectors[:, :n_factors] * np.sqrt(
            np.maximum(eigenvalues[:n_factors], 0)
        )

        # Uniqueness (specific variance)
        psi = np.diag(cov) - np.sum(L ** 2, axis=1)
        psi = np.maximum(psi, 1e-10)

        # Allocate inversely to uniqueness (favor well-explained assets)
        inv_psi = 1.0 / psi
        weights = inv_psi / inv_psi.sum()
        return weights
