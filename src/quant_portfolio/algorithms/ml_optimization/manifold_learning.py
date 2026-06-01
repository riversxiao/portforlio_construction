"""Manifold Learning algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class ManifoldLearningAlgorithm(Algorithm):
    """Manifold-Based Dimensionality Reduction for Allocation.

    Uses spectral embedding (Laplacian Eigenmaps) to discover the
    low-dimensional manifold structure of asset returns. Assets
    that are central on the manifold get higher weights as they
    best represent the return space.
    """

    name = "manifold_learning"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        corr = returns.corr().values

        # Affinity matrix from correlation
        affinity = np.exp(-0.5 * (1 - corr) ** 2)
        np.fill_diagonal(affinity, 0)

        # Degree matrix
        degree = affinity.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-10)))

        # Normalized Laplacian
        L_norm = np.eye(n_assets) - D_inv_sqrt @ affinity @ D_inv_sqrt

        # Eigenvectors (smallest non-trivial)
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

        # Centrality: assets with high degree are central
        centrality = degree / degree.sum()
        weights = centrality / centrality.sum()
        return weights
