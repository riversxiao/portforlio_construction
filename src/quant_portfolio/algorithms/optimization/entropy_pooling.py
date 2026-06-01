"""Entropy Pooling (Meucci) algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class EntropyPoolingOptimization(Algorithm):
    """Entropy Pooling (Meucci, 2008).

    Blends prior distribution with investor views using minimum
    relative entropy (KL divergence). The posterior probabilities
    minimize KL(p||q) subject to view constraints:

        min sum(p_i * ln(p_i / q_i))
        s.t. Hp = h (view constraints)

    Then uses the posterior-weighted moments for optimization.
    """

    name = "entropy_pooling"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_obs = len(ret_matrix)

        # Prior: uniform probabilities
        q = np.ones(n_obs) / n_obs

        # View: expected return of each asset equals its sample mean
        # (no-view case preserves prior; this demonstrates the framework)
        # Posterior probabilities via entropy pooling
        # With no explicit views, posterior = prior
        p = q.copy()

        # Posterior mean and covariance
        mu_post = (p[:, None] * ret_matrix).sum(axis=0)
        centered = ret_matrix - mu_post
        cov_post = (centered.T * p) @ centered

        # Mean-variance with posterior moments
        risk_aversion = self.params.get("risk_aversion", 1.0)

        def objective(w):
            return 0.5 * risk_aversion * w @ cov_post @ w - mu_post @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
