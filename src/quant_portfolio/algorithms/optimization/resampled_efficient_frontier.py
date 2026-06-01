"""Resampled Efficient Frontier (Michaud) algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class ResampledEfficientFrontierOptimization(Algorithm):
    """Resampled Efficient Frontier (Michaud, 1998).

    Addresses estimation error in mean-variance by resampling:
    1. Generate N bootstrap samples of returns
    2. For each sample, compute the optimal mean-variance portfolio
    3. Average the resulting weights across all samples

    This produces more stable, diversified portfolios that are less
    sensitive to estimation errors in expected returns.
    """

    name = "resampled_efficient_frontier"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        n_samples = self.params.get("n_samples", 100)
        risk_aversion = self.params.get("risk_aversion", 1.0)
        rng = np.random.default_rng(42)

        all_weights = []
        ret_values = returns.values
        n_obs = len(ret_values)

        for _ in range(n_samples):
            # Bootstrap sample
            idx = rng.integers(0, n_obs, size=n_obs)
            sample = ret_values[idx]
            mu_s = sample.mean(axis=0)
            cov_s = np.cov(sample, rowvar=False)

            def objective(w, mu=mu_s, cov=cov_s):
                return 0.5 * risk_aversion * w @ cov @ w - mu @ w

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            bounds = [(0, 1)] * n_assets
            w0 = np.ones(n_assets) / n_assets

            result = minimize(objective, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints)
            if result.success:
                all_weights.append(result.x)

        if all_weights:
            return np.mean(all_weights, axis=0)
        return np.ones(n_assets) / n_assets
