"""CVaR (Expected Shortfall) Calculation algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class CVaRCalculationAlgorithm(Algorithm):
    """Expected Shortfall / CVaR Risk-Adjusted Weights.

    Computes CVaR for each asset and allocates inversely proportional
    to individual asset CVaR, producing risk-adjusted weights that
    penalize assets with higher tail risk:

        w_i = (1/CVaR_i) / sum(1/CVaR_j)
    """

    name = "cvar_calculation"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        alpha = self.params.get("alpha", 0.05)
        ret_matrix = returns.values

        # Compute CVaR for each asset
        cvars = np.zeros(n_assets)
        for i in range(n_assets):
            sorted_ret = np.sort(ret_matrix[:, i])
            n_tail = max(1, int(len(sorted_ret) * alpha))
            cvars[i] = -sorted_ret[:n_tail].mean()

        # Avoid division by zero
        cvars = np.maximum(cvars, 1e-10)
        inv_cvar = 1.0 / cvars
        weights = inv_cvar / inv_cvar.sum()
        return weights
