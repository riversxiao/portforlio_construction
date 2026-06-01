"""Independent Component Analysis algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class IndependentComponentAlgorithm(Algorithm):
    """Independent Component Analysis (ICA) for Allocation.

    Uses ICA via the FastICA algorithm (fixed-point iteration with
    negentropy approximation) to find statistically independent
    sources in asset returns. Allocates based on the mixing matrix
    to achieve maximum independence of portfolio components.
    """

    name = "independent_component"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_components = min(self.params.get("n_components", n_assets), n_assets)

        # Center and whiten
        X = ret_matrix - ret_matrix.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        D = np.diag(1.0 / np.sqrt(np.maximum(eigenvalues[idx], 1e-10)))
        P = eigenvectors[:, idx]
        X_white = (X @ P @ D.T)

        # FastICA with tanh nonlinearity
        rng = np.random.default_rng(42)
        W = rng.normal(size=(n_components, n_components))
        # Orthogonalize
        W, _ = np.linalg.qr(W)

        for _ in range(50):
            # g(u) = tanh(u), g'(u) = 1 - tanh(u)^2
            WX = W @ X_white.T  # (n_comp, n_obs)
            tanh_WX = np.tanh(WX)
            W_new = (tanh_WX @ X_white) / X_white.shape[0] - \
                    (1 - tanh_WX ** 2).mean(axis=1, keepdims=True) * W
            W_new, _ = np.linalg.qr(W_new.T)
            W_new = W_new.T
            if np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1)) < 1e-6:
                W = W_new
                break
            W = W_new

        # Unmixing matrix in original space
        unmixing = W @ D @ P.T
        # Weight inversely to total unmixing magnitude
        mixing_strength = np.sqrt((unmixing ** 2).sum(axis=0))
        mixing_strength = np.maximum(mixing_strength, 1e-10)
        inv_strength = 1.0 / mixing_strength
        weights = inv_strength / inv_strength.sum()
        return weights
