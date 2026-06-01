"""Hierarchical Risk Parity (HRP) algorithm."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from quant_portfolio.algorithms.base import Algorithm


class HierarchicalRiskParityOptimization(Algorithm):
    """Hierarchical Risk Parity (HRP) by Marcos Lopez de Prado.

    Uses hierarchical clustering on the correlation matrix to determine
    portfolio structure, then allocates using a recursive bisection approach
    based on inverse variance within clusters.

    Steps:
    1. Compute distance matrix from correlations
    2. Hierarchical clustering (single linkage)
    3. Quasi-diagonalize the covariance matrix
    4. Recursive bisection for weight allocation
    """

    name = "hierarchical_risk_parity"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values
        corr = returns.corr().values

        # Distance matrix from correlation
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)
        dist = (dist + dist.T) / 2
        condensed = squareform(dist, checks=False)

        # Hierarchical clustering
        link = linkage(condensed, method="single")
        sort_ix = leaves_list(link).tolist()

        # Recursive bisection
        weights = np.zeros(n_assets)
        self._recursive_bisection(weights, sort_ix, cov)
        return weights

    def _recursive_bisection(self, weights, indices, cov):
        if len(indices) == 1:
            weights[indices[0]] = 1.0
            return

        mid = len(indices) // 2
        left = indices[:mid]
        right = indices[mid:]

        # Variance of each sub-cluster (inverse variance allocation)
        var_left = self._cluster_variance(left, cov)
        var_right = self._cluster_variance(right, cov)

        alpha = 1.0 - var_left / (var_left + var_right) if (var_left + var_right) > 1e-10 else 0.5

        # Allocate
        left_weights = np.zeros(len(weights))
        right_weights = np.zeros(len(weights))
        self._recursive_bisection(left_weights, left, cov)
        self._recursive_bisection(right_weights, right, cov)

        for i in left:
            weights[i] = alpha * left_weights[i]
        for i in right:
            weights[i] = (1.0 - alpha) * right_weights[i]

    def _cluster_variance(self, indices, cov):
        sub_cov = cov[np.ix_(indices, indices)]
        n = len(indices)
        w = np.ones(n) / n
        return w @ sub_cov @ w
