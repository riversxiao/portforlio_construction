"""Core-Satellite Allocation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class CoreSatelliteAlgorithm(Algorithm):
    """Core-Satellite Allocation.

    Divides the portfolio into a core holding (low-risk, diversified)
    and satellite positions (higher-risk, alpha-seeking):

    - Core: minimum variance weighted among lower-vol assets
    - Satellite: momentum-weighted among higher-vol assets

    Core proportion is configurable (default 70%).
    """

    name = "core_satellite"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        core_pct = self.params.get("core_pct", 0.7)
        ret_matrix = returns.values
        vols = returns.std().values

        # Split into core and satellite
        median_vol = np.median(vols)
        core_mask = vols <= median_vol
        sat_mask = ~core_mask

        weights = np.zeros(n_assets)
        n_core = core_mask.sum()
        n_sat = sat_mask.sum()

        if n_core > 0:
            # Core: inverse variance
            core_vols = vols[core_mask]
            inv_var = 1.0 / np.maximum(core_vols ** 2, 1e-10)
            core_w = inv_var / inv_var.sum()
            weights[core_mask] = core_pct * core_w

        if n_sat > 0:
            # Satellite: momentum-based
            cum_ret = (1 + ret_matrix[-min(63, len(ret_matrix)):]).prod(axis=0) - 1
            sat_scores = np.maximum(cum_ret[sat_mask], 0)
            total_sat = sat_scores.sum()
            if total_sat > 1e-10:
                weights[sat_mask] = (1 - core_pct) * sat_scores / total_sat
            else:
                weights[sat_mask] = (1 - core_pct) / n_sat

        if n_core == 0 or n_sat == 0:
            weights = np.ones(n_assets) / n_assets

        return weights
