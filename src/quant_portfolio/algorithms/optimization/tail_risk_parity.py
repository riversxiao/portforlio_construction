"""Tail Risk Parity algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class TailRiskParityOptimization(Algorithm):
    """Tail Risk Parity.

    Equalizes tail risk contributions across assets. Instead of using
    volatility (as in standard risk parity), uses CVaR contributions:

        TRC_i = w_i * partial CVaR / partial w_i

    Each asset contributes equally to the portfolio's tail risk (CVaR).
    """

    name = "tail_risk_parity"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        alpha = self.params.get("alpha", 0.05)

        def tail_risk_contribution(w):
            port_ret = ret_matrix @ w
            sorted_idx = np.argsort(port_ret)
            n_tail = max(1, int(len(port_ret) * alpha))
            tail_idx = sorted_idx[:n_tail]
            # Marginal CVaR contribution per asset
            tail_returns = ret_matrix[tail_idx]
            marginal_cvar = -tail_returns.mean(axis=0)
            trc = w * marginal_cvar
            return trc

        def objective(w):
            trc = tail_risk_contribution(w)
            total = trc.sum()
            if total < 1e-10:
                return 0.0
            target = total / n_assets
            return np.sum((trc - target) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.001, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
