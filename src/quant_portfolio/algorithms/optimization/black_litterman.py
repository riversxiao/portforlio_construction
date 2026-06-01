"""Black-Litterman Model algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class BlackLittermanOptimization(Algorithm):
    """Black-Litterman Model for Portfolio Optimization.

    Combines market equilibrium returns (implied by market cap weights)
    with investor views to produce a posterior expected return vector.
    The posterior is then used for mean-variance optimization.

    Posterior: mu_BL = [(tau*Sigma)^{-1} + P^T*Omega^{-1}*P]^{-1}
               * [(tau*Sigma)^{-1}*Pi + P^T*Omega^{-1}*Q]

    Where Pi = equilibrium returns, P = view matrix, Q = view returns,
    Omega = view uncertainty, tau = scaling factor.
    """

    name = "black_litterman"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values
        tau = self.params.get("tau", 0.05)
        risk_aversion = self.params.get("risk_aversion", 2.5)

        # Equilibrium weights (equal weight as proxy for market cap)
        w_mkt = np.ones(n_assets) / n_assets
        # Implied equilibrium returns
        pi = risk_aversion * cov @ w_mkt

        # Without explicit views, use equilibrium returns
        # Posterior mean = pi (no views case)
        mu_bl = pi

        # Optimize with posterior returns
        inv_cov = np.linalg.pinv(cov)
        w_star = (1.0 / risk_aversion) * inv_cov @ mu_bl
        # Normalize to sum to 1 and enforce non-negative
        w_star = np.maximum(w_star, 0)
        w_sum = w_star.sum()
        if w_sum > 1e-10:
            w_star = w_star / w_sum
        else:
            w_star = np.ones(n_assets) / n_assets
        return w_star
