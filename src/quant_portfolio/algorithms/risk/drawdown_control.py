"""Drawdown Control algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class DrawdownControlAlgorithm(Algorithm):
    """Drawdown Control Portfolio Optimization.

    Finds weights that minimize maximum drawdown of the portfolio.
    Computes cumulative returns for each candidate allocation and
    minimizes the peak-to-trough decline:

        MaxDD = max_t (peak_t - value_t) / peak_t
    """

    name = "drawdown_control"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values

        def max_drawdown(w):
            port_ret = ret_matrix @ w
            cum_ret = np.cumprod(1 + port_ret)
            peak = np.maximum.accumulate(cum_ret)
            drawdown = (peak - cum_ret) / peak
            return drawdown.max()

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(max_drawdown, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
