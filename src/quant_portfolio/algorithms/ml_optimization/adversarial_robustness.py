"""Adversarial Robustness algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class AdversarialRobustnessAlgorithm(Algorithm):
    """Adversarial Robustness in Portfolio Optimization.

    Finds weights that are robust to adversarial perturbations of the
    return distribution. Solves a minimax problem:

        min_w max_delta ||delta||<=epsilon : risk(w, mu+delta)

    Approximated by optimizing under worst-case mean perturbation
    within an epsilon-ball.
    """

    name = "adversarial_robustness"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        epsilon = self.params.get("epsilon", 0.001)
        risk_aversion = self.params.get("risk_aversion", 1.0)

        def robust_obj(w):
            # Worst-case perturbation direction: opposite to w
            w_norm = np.linalg.norm(w)
            if w_norm > 1e-10:
                delta = -epsilon * w / w_norm
            else:
                delta = np.zeros(n_assets)
            # Worst-case return
            worst_mu = mu + delta
            port_ret = worst_mu @ w
            port_var = w @ cov @ w
            return -port_ret + 0.5 * risk_aversion * port_var

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(robust_obj, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
