"""Risk Budgeting algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class RiskBudgetingAlgorithm(Algorithm):
    """Risk Budgeting Across Assets.

    Allocates weights such that each asset's risk contribution matches
    a specified risk budget. The risk contribution of asset i is:

        RC_i = w_i * (Sigma * w)_i / sigma_p

    The budget can be equal (risk parity) or custom per asset.
    """

    name = "risk_budgeting"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values
        # Default: equal risk budget
        budget = self.params.get("budget", np.ones(n_assets) / n_assets)
        budget = np.array(budget)
        budget = budget / budget.sum()

        def objective(w):
            port_var = w @ cov @ w
            if port_var < 1e-10:
                return 0.0
            port_vol = np.sqrt(port_var)
            marginal = cov @ w
            rc = w * marginal / port_vol
            rc_pct = rc / rc.sum() if rc.sum() > 1e-10 else rc
            return np.sum((rc_pct - budget) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.001, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
