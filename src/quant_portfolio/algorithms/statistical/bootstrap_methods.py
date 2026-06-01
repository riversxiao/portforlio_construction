"""Bootstrap Methods algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class BootstrapMethodsAlgorithm(Algorithm):
    """Bootstrap Confidence Intervals for Robust Allocation.

    Uses the bootstrap to estimate uncertainty in optimal weights.
    Generates B bootstrap samples, computes optimal weights for each,
    and returns the average, providing more robust estimates than a
    single sample optimization.
    """

    name = "bootstrap_methods"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_obs = len(ret_matrix)
        n_bootstrap = self.params.get("n_bootstrap", 50)
        rng = np.random.default_rng(42)

        all_weights = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_obs, size=n_obs)
            sample = ret_matrix[idx]
            mu_s = sample.mean(axis=0)
            cov_s = np.cov(sample, rowvar=False)

            # Min variance for each bootstrap sample
            def obj(w, cov=cov_s):
                return w @ cov @ w

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            bounds = [(0, 1)] * n_assets
            w0 = np.ones(n_assets) / n_assets
            result = minimize(obj, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints)
            if result.success:
                all_weights.append(result.x)

        if all_weights:
            return np.mean(all_weights, axis=0)
        return np.ones(n_assets) / n_assets
