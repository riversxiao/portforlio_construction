"""Graph-Based Allocation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class GraphBasedAlgorithm(Algorithm):
    """Graph/Network-Based Asset Allocation.

    Constructs an asset correlation network and uses network centrality
    measures to determine allocation. Assets with low centrality
    (peripheral nodes) provide more diversification.

    Uses eigenvector centrality: principal eigenvector of the
    adjacency matrix.
    """

    name = "graph_based"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        corr = returns.corr().values
        threshold = self.params.get("corr_threshold", 0.3)

        # Build adjacency matrix (threshold correlations)
        adj = (np.abs(corr) > threshold).astype(float)
        np.fill_diagonal(adj, 0)

        # Eigenvector centrality: dominant eigvec of adjacency
        eigenvalues, eigenvectors = np.linalg.eigh(adj)
        centrality = np.abs(eigenvectors[:, -1])

        # Allocate inversely to centrality (peripheral = diversifier)
        inv_centrality = 1.0 / np.maximum(centrality, 1e-10)
        weights = inv_centrality / inv_centrality.sum()
        return weights
