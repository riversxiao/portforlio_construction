"""Strategic Asset Allocation algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class StrategicAllocationAlgorithm(Algorithm):
    """Strategic Asset Allocation.

    Determines long-term optimal weights based on long-run equilibrium
    expected returns and risk. Uses a mean-variance framework with
    annualized estimates to find the policy portfolio:

        max mu^T * w - (lambda/2) * w^T * Sigma * w

    Annualizes daily returns for strategic horizon estimation.
    """

    name = "strategic_allocation"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        # Annualize
        mu = returns.mean().values * 252
        cov = returns.cov().values * 252
        risk_aversion = self.params.get("risk_aversion", 2.0)

        def objective(w):
            return -(mu @ w - 0.5 * risk_aversion * w @ cov @ w)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
