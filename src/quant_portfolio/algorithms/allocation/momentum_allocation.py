"""Momentum-Based Allocation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class MomentumAllocationAlgorithm(Algorithm):
    """Momentum-Based Asset Allocation.

    Allocates to assets based on their recent momentum (cumulative
    return over a lookback period). Assets with positive momentum
    receive proportionally higher weights:

        score_i = cumulative_return_i(lookback)
        w_i = max(score_i, 0) / sum(max(score_j, 0))
    """

    name = "momentum_allocation"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        lookback = self.params.get("lookback", 63)
        ret_matrix = returns.values

        window = min(lookback, len(ret_matrix))
        recent = ret_matrix[-window:]

        # Cumulative return as momentum score
        cum_ret = (1 + recent).prod(axis=0) - 1
        scores = np.maximum(cum_ret, 0)

        total = scores.sum()
        if total > 1e-10:
            weights = scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
