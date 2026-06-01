"""Transfer Learning algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class TransferLearningAlgorithm(Algorithm):
    """Transfer Learning for Domain Adaptation in Allocation.

    Transfers knowledge from a source domain (full history) to the
    target domain (recent period) via importance weighting. Older
    observations are downweighted based on distribution shift:

        weight_t = p_target(x_t) / p_source(x_t)

    Approximated by exponential decay from source to target period.
    """

    name = "transfer_learning"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_obs = len(ret_matrix)
        decay = self.params.get("decay", 0.02)

        # Importance weights: exponential decay
        time_weights = np.exp(decay * (np.arange(n_obs) - n_obs))
        time_weights = time_weights / time_weights.sum()

        # Weighted mean and covariance
        mu_w = (time_weights[:, None] * ret_matrix).sum(axis=0)
        centered = ret_matrix - mu_w
        cov_w = (centered.T * time_weights) @ centered

        # Mean-variance with transferred estimates
        risk_aversion = self.params.get("risk_aversion", 1.0)

        def objective(w):
            return 0.5 * risk_aversion * w @ cov_w @ w - mu_w @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
