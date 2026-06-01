"""Target Date (Lifecycle) Allocation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class TargetDateAlgorithm(Algorithm):
    """Target Date / Lifecycle Glide Path Allocation.

    Implements a glide path that reduces risk exposure over time.
    Uses the ratio of elapsed time to total horizon to determine
    the equity/risk allocation:

        equity_pct = max_equity * (1 - elapsed / horizon)

    Higher risk assets (higher volatility) are treated as equity-like,
    lower risk assets as bond-like.
    """

    name = "target_date"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        n_obs = len(returns)
        horizon = self.params.get("horizon", 252 * 10)  # 10 years
        max_equity = self.params.get("max_equity", 0.9)

        # Progress along glide path
        elapsed_ratio = min(n_obs / horizon, 1.0)
        equity_pct = max_equity * (1 - elapsed_ratio)

        # Classify assets by volatility
        vols = returns.std().values
        median_vol = np.median(vols)

        equity_mask = vols >= median_vol
        bond_mask = ~equity_mask

        weights = np.zeros(n_assets)
        n_equity = equity_mask.sum()
        n_bond = bond_mask.sum()

        if n_equity > 0:
            weights[equity_mask] = equity_pct / n_equity
        if n_bond > 0:
            weights[bond_mask] = (1 - equity_pct) / n_bond

        if n_equity == 0 or n_bond == 0:
            weights = np.ones(n_assets) / n_assets

        return weights
