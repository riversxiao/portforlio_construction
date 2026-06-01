"""Factor Risk Decomposition algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class FactorRiskDecompositionAlgorithm(Algorithm):
    """Factor Risk Decomposition Allocation.

    Decomposes total risk into factor (systematic) and specific
    (idiosyncratic) components using PCA. Allocates to minimize
    concentration in any single factor:

    1. Extract principal components from returns
    2. Compute factor loadings for each asset
    3. Weight inversely to total factor exposure magnitude
    """

    name = "factor_risk_decomposition"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values
        n_factors = min(self.params.get("n_factors", 3), n_assets)

        # PCA for factor decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx][:, :n_factors]
        eigenvalues = eigenvalues[idx][:n_factors]

        # Factor loadings (beta to each factor)
        loadings = eigenvectors * np.sqrt(eigenvalues)

        # Total factor exposure per asset
        factor_exposure = np.sqrt((loadings ** 2).sum(axis=1))
        factor_exposure = np.maximum(factor_exposure, 1e-10)

        # Inverse factor exposure weighting
        inv_exp = 1.0 / factor_exposure
        weights = inv_exp / inv_exp.sum()
        return weights
