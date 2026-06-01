"""Autoencoder Feature Extraction algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class AutoencoderFeaturesAlgorithm(Algorithm):
    """Autoencoder for Feature Extraction and Allocation.

    Uses a simple linear autoencoder to compress the return matrix
    into a lower-dimensional representation. The reconstruction
    quality per asset indicates how well it fits the common structure.
    Allocates based on reconstruction residual (lower = more systematic).
    """

    name = "autoencoder_features"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_components = min(self.params.get("n_components", 3), n_assets)
        n_epochs = self.params.get("n_epochs", 50)
        lr = self.params.get("lr", 0.01)
        rng = np.random.default_rng(42)

        # Standardize
        X = (ret_matrix - ret_matrix.mean(axis=0)) / (ret_matrix.std(axis=0) + 1e-10)

        # Linear autoencoder: W_enc (n_assets -> n_comp), W_dec (n_comp -> n_assets)
        W_enc = rng.normal(0, 0.1, (n_assets, n_components))
        W_dec = rng.normal(0, 0.1, (n_components, n_assets))

        for _ in range(n_epochs):
            # Forward
            encoded = X @ W_enc
            decoded = encoded @ W_dec
            error = decoded - X

            # Gradient
            grad_dec = encoded.T @ error / len(X)
            grad_enc = X.T @ (error @ W_dec.T) / len(X)

            W_dec -= lr * grad_dec
            W_enc -= lr * grad_enc

        # Reconstruction error per asset
        encoded = X @ W_enc
        decoded = encoded @ W_dec
        recon_error = np.mean((decoded - X) ** 2, axis=0)

        # Low reconstruction error = well explained = allocate more
        inv_error = 1.0 / np.maximum(recon_error, 1e-10)
        weights = inv_error / inv_error.sum()
        return weights
