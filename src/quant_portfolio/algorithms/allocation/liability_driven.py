"""Liability-Driven Investment algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class LiabilityDrivenAlgorithm(Algorithm):
    """Liability-Driven Investment (LDI) Allocation.

    Optimizes the portfolio to hedge a liability stream while seeking
    surplus growth. The liability is modeled as the minimum acceptable
    return. The objective minimizes tracking error to the liability:

        min Var(R_p - R_L)

    where R_L is the liability return (approximated by the lowest
    volatility asset or a target return).
    """

    name = "liability_driven"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values
        target_return = self.params.get("liability_return", 0.0001)
        mu = returns.mean().values

        # Minimize variance subject to meeting liability return
        def objective(w):
            return w @ cov @ w

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: mu @ w - target_return},
        ]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
