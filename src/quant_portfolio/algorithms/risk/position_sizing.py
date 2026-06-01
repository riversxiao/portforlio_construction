"""Optimal Position Sizing algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class PositionSizingAlgorithm(Algorithm):
    """Optimal Position Sizing using volatility-adjusted Kelly.

    Combines Kelly criterion with volatility targeting to determine
    optimal position sizes. Each asset's position is sized by:

        size_i = (mu_i / sigma_i^2) * target_vol / portfolio_vol

    Weights are then normalized to sum to 1.
    """

    name = "position_sizing"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        var = returns.var().values
        target_vol = self.params.get("target_vol", 0.15 / np.sqrt(252))

        # Kelly-based sizing
        var_safe = np.maximum(var, 1e-10)
        raw_sizes = mu / var_safe

        # Remove negative sizes (short positions) for long-only
        raw_sizes = np.maximum(raw_sizes, 0)
        total = raw_sizes.sum()
        if total > 1e-10:
            weights = raw_sizes / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
