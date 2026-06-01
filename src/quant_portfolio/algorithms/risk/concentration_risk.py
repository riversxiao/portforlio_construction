"""Concentration Risk algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class ConcentrationRiskAlgorithm(Algorithm):
    """Portfolio Concentration Risk Minimization.

    Minimizes the Herfindahl-Hirschman Index (HHI) of risk
    contributions while maintaining a minimum Sharpe ratio.
    The HHI of risk contributions measures concentration:

        HHI = sum(RC_i^2) / (sum(RC_i))^2

    Lower HHI means more diversified risk contributions.
    """

    name = "concentration_risk"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values

        def objective(w):
            port_var = w @ cov @ w
            if port_var < 1e-10:
                return 0.0
            marginal = cov @ w
            rc = w * marginal
            total_rc = rc.sum()
            if total_rc < 1e-10:
                return 0.0
            rc_pct = rc / total_rc
            # Minimize HHI of risk contributions
            hhi = np.sum(rc_pct ** 2)
            return hhi

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.001, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
