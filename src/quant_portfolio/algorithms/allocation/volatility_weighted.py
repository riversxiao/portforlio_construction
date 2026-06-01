"""Volatility-Weighted Allocation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class VolatilityWeightedAlgorithm(Algorithm):
    """Volatility-Weighted Allocation.

    Allocates inversely proportional to realized volatility:

        w_i = (1/sigma_i) / sum(1/sigma_j)

    This simple heuristic ensures equal volatility contribution
    from each asset (approximately), without requiring correlation
    estimates.
    """

    name = "volatility_weighted"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        vols = returns.std().values
        vols = np.maximum(vols, 1e-10)
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        return weights
