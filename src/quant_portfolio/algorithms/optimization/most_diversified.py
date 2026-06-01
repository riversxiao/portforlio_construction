"""Most Diversified Portfolio algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class MostDiversifiedOptimization(Algorithm):
    """Most Diversified Portfolio.

    Maximizes the ratio of weighted average volatility to portfolio
    volatility (diversification ratio), equivalent to minimizing
    portfolio variance given volatility-scaled weights.

        DR = sum(w_i * sigma_i) / sigma_p

    This targets maximum benefit from diversification.
    """

    name = "most_diversified"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values
        vols = np.sqrt(np.diag(cov))

        def neg_div_ratio(w):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 0.0
            weighted_vol = w @ vols
            return -weighted_vol / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(neg_div_ratio, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
