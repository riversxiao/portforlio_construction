"""Omega Ratio Maximization algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class OmegaRatioOptimization(Algorithm):
    """Omega Ratio Maximization.

    The Omega ratio is the ratio of gains to losses relative to a threshold:

        Omega(L) = E[max(R - L, 0)] / E[max(L - R, 0)]

    where L is the threshold return. Maximizing Omega captures the full
    return distribution rather than just mean and variance.
    """

    name = "omega_ratio"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        threshold = self.params.get("threshold", 0.0)

        def neg_omega(w):
            port_returns = ret_matrix @ w
            gains = np.maximum(port_returns - threshold, 0).mean()
            losses = np.maximum(threshold - port_returns, 0).mean()
            if losses < 1e-10:
                return -100.0
            return -(gains / losses)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(neg_omega, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
