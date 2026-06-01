"""Tactical Asset Allocation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class TacticalAllocationAlgorithm(Algorithm):
    """Tactical Asset Allocation.

    Short-term tilts around strategic weights based on momentum signals.
    Uses recent performance relative to long-term average to determine
    tactical overweight/underweight:

        tilt_i = (recent_return_i - long_avg_i) / vol_i
        w_i = strategic_w_i * (1 + scale * tilt_i)
    """

    name = "tactical_allocation"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        lookback_short = self.params.get("lookback_short", 21)
        lookback_long = self.params.get("lookback_long", 252)
        scale = self.params.get("tilt_scale", 0.2)
        ret_matrix = returns.values

        # Strategic base: equal weight
        base_w = np.ones(n_assets) / n_assets

        # Tactical tilts
        n_obs = len(ret_matrix)
        short_window = min(lookback_short, n_obs)
        long_window = min(lookback_long, n_obs)

        short_ret = ret_matrix[-short_window:].mean(axis=0)
        long_ret = ret_matrix[-long_window:].mean(axis=0)
        vols = ret_matrix.std(axis=0)
        vols = np.maximum(vols, 1e-10)

        tilts = (short_ret - long_ret) / vols
        weights = base_w * (1 + scale * tilts)
        weights = np.maximum(weights, 0)
        total = weights.sum()
        if total > 1e-10:
            weights = weights / total
        else:
            weights = base_w
        return weights
