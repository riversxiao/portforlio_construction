"""Regime-Based Allocation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class RegimeBasedAllocationAlgorithm(Algorithm):
    """Regime-Based Conditional Allocation.

    Detects market regimes using rolling volatility and correlation,
    then applies regime-specific allocation rules:

    - Bull (low vol, positive returns): momentum-weighted
    - Bear (high vol, negative returns): minimum variance
    - Neutral: equal weight
    """

    name = "regime_based_allocation"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        lookback = self.params.get("lookback", 63)

        window = min(lookback, len(ret_matrix))
        recent = ret_matrix[-window:]
        recent_mean = recent.mean(axis=1).mean()
        recent_vol = recent.mean(axis=1).std()
        long_vol = ret_matrix.mean(axis=1).std()

        vols = returns.std().values

        if recent_mean > 0 and recent_vol < long_vol:
            # Bull regime: momentum weights
            cum_ret = (1 + recent).prod(axis=0) - 1
            scores = np.maximum(cum_ret, 0)
            total = scores.sum()
            if total > 1e-10:
                weights = scores / total
            else:
                weights = np.ones(n_assets) / n_assets
        elif recent_mean < 0 or recent_vol > 1.5 * long_vol:
            # Bear regime: inverse volatility
            inv_vol = 1.0 / np.maximum(vols, 1e-10)
            weights = inv_vol / inv_vol.sum()
        else:
            # Neutral: equal weight
            weights = np.ones(n_assets) / n_assets

        return weights
