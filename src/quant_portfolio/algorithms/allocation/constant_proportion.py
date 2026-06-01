"""Constant Proportion Portfolio Insurance (CPPI) algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class ConstantProportionAlgorithm(Algorithm):
    """Constant Proportion Portfolio Insurance (CPPI).

    Maintains risky allocation as a multiple of the cushion
    (portfolio value minus floor):

        risky_allocation = m * (V - F) / V

    where m is the multiplier, V is portfolio value, and F is the floor.
    The rest goes to the safe asset (lowest volatility asset).
    """

    name = "constant_proportion"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        multiplier = self.params.get("multiplier", 3.0)
        floor_pct = self.params.get("floor_pct", 0.8)
        ret_matrix = returns.values

        # Compute current portfolio value assuming unit start
        cum_ret = (1 + ret_matrix.mean(axis=1)).cumprod()
        current_value = cum_ret[-1] if len(cum_ret) > 0 else 1.0
        floor_value = floor_pct

        cushion = max(current_value - floor_value, 0) / current_value
        risky_pct = min(multiplier * cushion, 1.0)

        # Identify safe asset (lowest volatility)
        vols = returns.std().values
        safe_idx = np.argmin(vols)

        # Risky portion distributed by inverse vol among risky assets
        weights = np.zeros(n_assets)
        risky_vols = vols.copy()
        risky_vols[safe_idx] = np.inf
        inv_vol = 1.0 / np.maximum(risky_vols, 1e-10)
        inv_vol[safe_idx] = 0

        total_inv = inv_vol.sum()
        if total_inv > 1e-10:
            weights = risky_pct * inv_vol / total_inv
        weights[safe_idx] = 1.0 - risky_pct

        return weights
