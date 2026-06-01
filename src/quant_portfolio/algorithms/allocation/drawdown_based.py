"""Drawdown-Based Dynamic Allocation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class DrawdownBasedAlgorithm(Algorithm):
    """Drawdown-Based Dynamic Allocation.

    Adjusts allocation dynamically based on each asset's current
    drawdown from its peak. Assets in deep drawdown receive lower
    weights (trend following logic):

        w_i = (1 - dd_i / max_dd)^power / sum(...)

    where dd_i is current drawdown and max_dd is the threshold.
    """

    name = "drawdown_based"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        max_dd_threshold = self.params.get("max_dd_threshold", 0.20)
        power = self.params.get("power", 2.0)

        scores = np.zeros(n_assets)
        for i in range(n_assets):
            cum_ret = np.cumprod(1 + ret_matrix[:, i])
            peak = np.maximum.accumulate(cum_ret)
            dd = (peak[-1] - cum_ret[-1]) / peak[-1]
            ratio = min(dd / max_dd_threshold, 1.0)
            scores[i] = (1.0 - ratio) ** power

        total = scores.sum()
        if total > 1e-10:
            weights = scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
