"""Gaussian Process algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class GaussianProcessAlgorithm(Algorithm):
    """Gaussian Process Regression for Return Prediction.

    Uses GP regression with an RBF kernel to predict expected returns
    from recent patterns. The predictive mean gives the expected
    return estimate; the uncertainty is used to weight assets.

    Kernel: k(x,x') = sigma^2 * exp(-||x-x'||^2 / (2*l^2))
    """

    name = "gaussian_process"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_obs = len(ret_matrix)
        lookback = min(self.params.get("lookback", 30), n_obs - 1)

        if n_obs < 5:
            return np.ones(n_assets) / n_assets

        predicted_returns = np.zeros(n_assets)
        for i in range(n_assets):
            series = ret_matrix[:, i]
            # Use recent history as training
            X_train = np.arange(n_obs - lookback, n_obs - 1).reshape(-1, 1)
            y_train = series[-(lookback):-1]
            X_test = np.array([[n_obs - 1]])

            # RBF kernel
            l = self.params.get("length_scale", 5.0)
            sigma_f = series.std()
            noise = sigma_f * 0.1

            n_train = len(X_train)
            K = sigma_f ** 2 * np.exp(
                -0.5 * (X_train - X_train.T) ** 2 / l ** 2
            )
            K += noise ** 2 * np.eye(n_train)

            k_star = sigma_f ** 2 * np.exp(
                -0.5 * (X_test - X_train.T) ** 2 / l ** 2
            )

            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(n_train))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                predicted_returns[i] = (k_star @ alpha).item()
            except np.linalg.LinAlgError:
                predicted_returns[i] = series.mean()

        scores = np.maximum(predicted_returns, 0)
        total = scores.sum()
        if total > 1e-10:
            weights = scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
