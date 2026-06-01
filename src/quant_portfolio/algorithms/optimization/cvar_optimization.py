"""CVaR (Conditional Value-at-Risk) Optimization algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class CVaROptimization(Algorithm):
    """CVaR (Conditional Value-at-Risk) Portfolio Optimization.

    Minimizes portfolio CVaR (Expected Shortfall) at a given confidence
    level alpha. CVaR is the expected loss in the worst (1-alpha) fraction
    of scenarios:

        CVaR_alpha = E[L | L >= VaR_alpha]

    Uses historical simulation: sort portfolio returns, average the worst
    (1-alpha) fraction.
    """

    name = "cvar_optimization"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        alpha = self.params.get("alpha", 0.05)

        def cvar_objective(w):
            port_returns = ret_matrix @ w
            sorted_returns = np.sort(port_returns)
            n_tail = max(1, int(len(sorted_returns) * alpha))
            cvar = -sorted_returns[:n_tail].mean()
            return cvar

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(cvar_objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
