"""Dynamic Stop-Loss algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class StopLossAlgorithm(Algorithm):
    """Dynamic Stop-Loss Weighting.

    Adjusts weights based on trailing drawdown from peak. Assets
    experiencing drawdowns exceeding a threshold have their weights
    reduced proportionally:

        w_i = base_w_i * max(0, 1 - drawdown_i / threshold)

    This implements a dynamic risk reduction mechanism.
    """

    name = "stop_loss"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        threshold = self.params.get("threshold", 0.10)

        # Compute trailing drawdown for each asset
        weights = np.zeros(n_assets)
        for i in range(n_assets):
            cum_ret = np.cumprod(1 + ret_matrix[:, i])
            peak = np.maximum.accumulate(cum_ret)
            current_dd = (peak[-1] - cum_ret[-1]) / peak[-1]
            # Scale weight by how far from stop-loss threshold
            scale = max(0.0, 1.0 - current_dd / threshold)
            weights[i] = scale

        total = weights.sum()
        if total > 1e-10:
            weights = weights / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
