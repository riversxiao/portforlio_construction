"""Goal-Based Allocation algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class GoalBasedAlgorithm(Algorithm):
    """Goal-Based Asset Allocation.

    Maximizes the probability of reaching a target wealth level
    within a given horizon. Uses the approximation:

        P(W_T >= G) ~ Phi((mu_p * T - log(G/W0)) / (sigma_p * sqrt(T)))

    Maximizes this probability by optimizing the risk-return trade-off
    relative to the goal.
    """

    name = "goal_based"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values * 252  # annualize
        cov = returns.cov().values * 252
        goal_return = self.params.get("goal_return", 0.08)

        def neg_goal_prob(w):
            port_mu = mu @ w
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 0.0
            # Maximize (mu - goal) / vol
            return -(port_mu - goal_return) / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(neg_goal_prob, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
