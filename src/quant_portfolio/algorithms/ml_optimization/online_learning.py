"""Online Learning algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class OnlineLearningAlgorithm(Algorithm):
    """Online / Incremental Learning for Portfolio Allocation.

    Implements the Online Newton Step (ONS) algorithm for
    portfolio selection. Updates weights sequentially as new
    data arrives using multiplicative updates:

        w_{t+1} = w_t * exp(eta * r_t) / sum(...)

    This is related to the Universal Portfolio algorithm.
    """

    name = "online_learning"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        eta = self.params.get("eta", 0.5)

        # Exponential gradient algorithm
        w = np.ones(n_assets) / n_assets

        for t in range(len(ret_matrix)):
            r_t = ret_matrix[t]
            # Multiplicative update
            w = w * np.exp(eta * r_t)
            # Normalize
            total = w.sum()
            if total > 1e-10:
                w = w / total
            else:
                w = np.ones(n_assets) / n_assets

        return w
