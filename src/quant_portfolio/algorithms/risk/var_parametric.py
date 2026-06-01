"""Parametric VaR algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from quant_portfolio.algorithms.base import Algorithm


class VaRParametricAlgorithm(Algorithm):
    """Parametric (Gaussian) VaR Portfolio Optimization.

    Assumes returns are normally distributed. Parametric VaR is:

        VaR_alpha = -(mu_p + z_alpha * sigma_p)

    where z_alpha is the alpha-quantile of the standard normal.
    Minimizes this parametric VaR estimate.
    """

    name = "var_parametric"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        alpha = self.params.get("alpha", 0.05)
        z = norm.ppf(alpha)

        def parametric_var(w):
            port_mu = mu @ w
            port_vol = np.sqrt(w @ cov @ w)
            return -(port_mu + z * port_vol)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(parametric_var, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
