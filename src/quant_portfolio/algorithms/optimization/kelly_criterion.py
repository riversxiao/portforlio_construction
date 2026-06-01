"""Kelly Criterion Portfolio Sizing algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class KellyCriterionOptimization(Algorithm):
    """Kelly Criterion for Optimal Portfolio Sizing.

    The Kelly criterion maximizes the expected log growth rate of wealth:

        max E[log(1 + w^T * r)]

    For the multivariate case with normally distributed returns:

        w* = Sigma^{-1} * mu / gamma

    where gamma is a fractional Kelly parameter (gamma=1 for full Kelly,
    gamma=2 for half Kelly which is more conservative).
    """

    name = "kelly_criterion"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        fraction = self.params.get("fraction", 0.5)  # Half-Kelly default

        try:
            inv_cov = np.linalg.pinv(cov)
            w = fraction * inv_cov @ mu
        except np.linalg.LinAlgError:
            w = np.ones(n_assets) / n_assets

        # Normalize: clip negative, normalize to sum to 1
        w = np.maximum(w, 0)
        w_sum = w.sum()
        if w_sum > 1e-10:
            w = w / w_sum
        else:
            w = np.ones(n_assets) / n_assets
        return w
