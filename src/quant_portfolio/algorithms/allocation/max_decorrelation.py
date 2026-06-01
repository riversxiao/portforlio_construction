"""Maximum Decorrelation Allocation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class MaxDecorrelationAlgorithm(Algorithm):
    """Maximum Decorrelation Portfolio.

    Allocates weights to maximize decorrelation, equivalent to
    the minimum variance portfolio using the correlation matrix
    instead of the covariance matrix:

        w* = Corr^{-1} * 1 / (1^T * Corr^{-1} * 1)

    This removes the influence of individual volatilities.
    """

    name = "max_decorrelation"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        corr = returns.corr().values

        try:
            inv_corr = np.linalg.pinv(corr)
            ones = np.ones(n_assets)
            w = inv_corr @ ones
            w = np.maximum(w, 0)
            total = w.sum()
            if total > 1e-10:
                w = w / total
            else:
                w = np.ones(n_assets) / n_assets
        except np.linalg.LinAlgError:
            w = np.ones(n_assets) / n_assets
        return w
