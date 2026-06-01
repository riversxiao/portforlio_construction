"""Minimum CVaR Portfolio algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class MinimumCVaROptimization(Algorithm):
    """Minimum CVaR (Conditional Value-at-Risk) Portfolio.

    Minimizes the expected shortfall at confidence level alpha.
    Unlike mean-CVaR which trades off return vs CVaR, this purely
    minimizes tail risk:

        min CVaR_alpha(w) = min E[-R_p | R_p <= VaR_alpha]
    """

    name = "minimum_cvar"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        alpha = self.params.get("alpha", 0.05)

        def cvar_obj(w):
            port_ret = ret_matrix @ w
            sorted_ret = np.sort(port_ret)
            n_tail = max(1, int(len(sorted_ret) * alpha))
            return -sorted_ret[:n_tail].mean()

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(cvar_obj, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
