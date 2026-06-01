"""Clustering-Based Allocation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class ClusteringAllocationAlgorithm(Algorithm):
    """K-Means Clustering-Based Allocation.

    Clusters assets based on return correlation, then allocates
    equally across clusters and within each cluster by inverse
    variance. This ensures diversification across groups of
    similar assets.
    """

    name = "clustering_allocation"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        n_clusters = min(self.params.get("n_clusters", 3), n_assets)

        # Simple K-means on return profiles
        ret_matrix = returns.values.T  # (n_assets, n_obs)
        rng = np.random.default_rng(42)

        # Initialize centroids
        idx = rng.choice(n_assets, n_clusters, replace=False)
        centroids = ret_matrix[idx].copy()

        labels = np.zeros(n_assets, dtype=int)
        for _ in range(20):
            # Assign
            for i in range(n_assets):
                dists = [np.linalg.norm(ret_matrix[i] - c) for c in centroids]
                labels[i] = np.argmin(dists)
            # Update centroids
            for k in range(n_clusters):
                members = ret_matrix[labels == k]
                if len(members) > 0:
                    centroids[k] = members.mean(axis=0)

        # Allocate: equal across clusters, inverse var within
        vols = returns.std().values
        weights = np.zeros(n_assets)
        for k in range(n_clusters):
            mask = labels == k
            n_members = mask.sum()
            if n_members == 0:
                continue
            cluster_vols = vols[mask]
            inv_var = 1.0 / np.maximum(cluster_vols ** 2, 1e-10)
            cluster_w = inv_var / inv_var.sum()
            weights[mask] = cluster_w / n_clusters

        total = weights.sum()
        if total > 1e-10:
            weights = weights / total
        return weights
