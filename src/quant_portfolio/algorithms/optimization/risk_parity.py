"""Risk Parity (Equal Risk Contribution) algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class RiskParityOptimization(Algorithm):
    """Risk Parity / Equal Risk Contribution Portfolio.

    Allocates weights such that each asset contributes equally to total
    portfolio risk. The risk contribution of asset i is:

        RC_i = w_i * (Sigma * w)_i / sqrt(w^T * Sigma * w)

    The objective minimizes the sum of squared differences between each
    asset's risk contribution and the target (1/N).
    """

    name = "risk_parity"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values
        target_risk = np.ones(n_assets) / n_assets

        def risk_contribution(w):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return np.ones(n_assets) / n_assets
            marginal = cov @ w
            rc = w * marginal / port_vol
            return rc

        def objective(w):
            rc = risk_contribution(w)
            rc_norm = rc / rc.sum() if rc.sum() > 1e-10 else rc
            return np.sum((rc_norm - target_risk) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.001, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
