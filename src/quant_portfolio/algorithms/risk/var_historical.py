"""Historical VaR algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class VaRHistoricalAlgorithm(Algorithm):
    """Historical Value-at-Risk Portfolio Optimization.

    Minimizes portfolio VaR using the historical simulation method.
    VaR at confidence level alpha is the alpha-quantile of the
    historical portfolio return distribution:

        VaR_alpha = -quantile(R_p, alpha)

    Returns weights that minimize the historical VaR.
    """

    name = "var_historical"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        alpha = self.params.get("alpha", 0.05)

        def var_obj(w):
            port_ret = ret_matrix @ w
            return -np.percentile(port_ret, alpha * 100)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(var_obj, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
