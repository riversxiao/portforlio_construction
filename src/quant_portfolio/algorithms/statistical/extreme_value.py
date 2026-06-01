"""Extreme Value Theory algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class ExtremeValueAlgorithm(Algorithm):
    """Extreme Value Theory (EVT) Based Allocation.

    Uses the Peaks-Over-Threshold (POT) approach from EVT to
    estimate tail behavior. The Generalized Pareto Distribution
    shape parameter (xi) indicates tail heaviness:

    - xi > 0: heavy tail (Frechet)
    - xi = 0: exponential tail (Gumbel)
    - xi < 0: bounded tail (Weibull)

    Allocates inversely to tail heaviness.
    """

    name = "extreme_value"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        threshold_pct = self.params.get("threshold_pct", 0.9)

        xi_estimates = np.zeros(n_assets)
        for i in range(n_assets):
            losses = -ret_matrix[:, i]
            threshold = np.percentile(losses, threshold_pct * 100)
            exceedances = losses[losses > threshold] - threshold

            if len(exceedances) < 5:
                xi_estimates[i] = 0.5
                continue

            # Method of moments estimator for GPD shape parameter
            mean_exc = exceedances.mean()
            var_exc = exceedances.var()
            if mean_exc > 1e-10:
                ratio = var_exc / (mean_exc ** 2)
                xi = 0.5 * (ratio - 1)
                xi_estimates[i] = max(xi, 0.01)
            else:
                xi_estimates[i] = 0.5

        # Inverse tail heaviness weighting
        inv_xi = 1.0 / np.maximum(xi_estimates, 0.01)
        weights = inv_xi / inv_xi.sum()
        return weights
