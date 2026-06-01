"""Stress Testing algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class StressTestingAlgorithm(Algorithm):
    """Stress Testing Portfolio Optimization.

    Constructs weights that minimize portfolio loss under stress
    scenarios. Uses the worst historical periods (tail events) as
    stress scenarios and optimizes to minimize worst-case loss:

        min max_s (loss under scenario s)

    Approximated by minimizing average loss in the worst quantile.
    """

    name = "stress_testing"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        stress_quantile = self.params.get("stress_quantile", 0.1)

        def stress_obj(w):
            port_ret = ret_matrix @ w
            sorted_ret = np.sort(port_ret)
            n_stress = max(1, int(len(sorted_ret) * stress_quantile))
            # Minimize worst-case average loss
            return -sorted_ret[:n_stress].mean()

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(stress_obj, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
