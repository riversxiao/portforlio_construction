"""Monte Carlo VaR algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class VaRMonteCarloAlgorithm(Algorithm):
    """Monte Carlo VaR Portfolio Optimization.

    Simulates portfolio returns using a multivariate normal distribution
    fitted to historical data, then computes VaR from the simulated
    distribution:

        1. Estimate mu, Sigma from historical returns
        2. Simulate N scenarios from N(mu, Sigma)
        3. VaR_alpha = -percentile(simulated portfolio returns, alpha)
    """

    name = "var_monte_carlo"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        alpha = self.params.get("alpha", 0.05)
        n_sims = self.params.get("n_simulations", 5000)
        rng = np.random.default_rng(42)

        # Pre-simulate scenarios
        scenarios = rng.multivariate_normal(mu, cov, size=n_sims)

        def mc_var(w):
            port_ret = scenarios @ w
            return -np.percentile(port_ret, alpha * 100)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(mc_var, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
