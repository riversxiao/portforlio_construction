"""Bayesian Estimation algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class BayesianEstimationAlgorithm(Algorithm):
    """Bayesian Parameter Estimation for Portfolio Optimization.

    Uses Bayesian shrinkage of expected returns toward the
    grand mean (James-Stein type estimator):

        mu_bayes = (1-B) * mu_sample + B * mu_grand

    where B is the shrinkage factor based on the dispersion of
    sample means relative to estimation uncertainty.
    """

    name = "bayesian_estimation"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        n_obs = len(returns)
        mu_sample = returns.mean().values
        cov = returns.cov().values

        # Grand mean (prior)
        mu_grand = mu_sample.mean()

        # James-Stein shrinkage factor
        var_means = np.var(mu_sample)
        avg_var = np.trace(cov) / (n_assets * n_obs)
        if var_means > avg_var:
            B = min((n_assets - 2) * avg_var / (var_means * n_assets), 1.0)
        else:
            B = 0.9

        mu_bayes = (1 - B) * mu_sample + B * mu_grand

        # Mean-variance with Bayesian estimates
        risk_aversion = self.params.get("risk_aversion", 1.0)

        def objective(w):
            return 0.5 * risk_aversion * w @ cov @ w - mu_bayes @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
