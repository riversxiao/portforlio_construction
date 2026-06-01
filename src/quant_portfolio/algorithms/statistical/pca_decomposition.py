"""PCA Decomposition algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class PCADecompositionAlgorithm(Algorithm):
    """PCA-Based Factor Extraction for Allocation.

    Performs Principal Component Analysis on the return covariance
    matrix to identify the dominant risk factors. Allocates by
    minimizing exposure to the first principal component (market
    factor) while maintaining diversification.

    The minimum PC1 exposure portfolio reduces systematic risk.
    """

    name = "pca_decomposition"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # First PC loadings (market factor)
        pc1 = eigenvectors[:, 0]

        # Allocate inversely to PC1 loading magnitude
        pc1_abs = np.abs(pc1)
        pc1_abs = np.maximum(pc1_abs, 1e-10)
        inv_loading = 1.0 / pc1_abs
        weights = inv_loading / inv_loading.sum()
        return weights
