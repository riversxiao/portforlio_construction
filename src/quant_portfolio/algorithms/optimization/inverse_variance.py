"""Inverse Variance Weighting algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class InverseVarianceOptimization(Algorithm):
    """Inverse Variance Weighting.

    Allocates weights inversely proportional to each asset's variance:

        w_i = (1/sigma_i^2) / sum(1/sigma_j^2)

    This is a simple diversification heuristic that tilts toward
    lower-volatility assets without requiring return estimates.
    """

    name = "inverse_variance"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        variances = returns.var().values
        # Avoid division by zero
        variances = np.maximum(variances, 1e-10)
        inv_var = 1.0 / variances
        weights = inv_var / inv_var.sum()
        return weights
