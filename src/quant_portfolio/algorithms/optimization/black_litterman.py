"""Black-Litterman Model algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class BlackLittermanOptimization(Algorithm):
    """Black-Litterman Model for Portfolio Optimization.

    Combines market equilibrium returns (implied by market cap weights)
    with investor views to produce a posterior expected return vector.
    The posterior is then used for mean-variance optimization.

    Posterior (with views):
        mu_BL = [(tau*Sigma)^{-1} + P^T*Omega^{-1}*P]^{-1}
                * [(tau*Sigma)^{-1}*Pi + P^T*Omega^{-1}*Q]

    Without views:
        mu_BL = Pi (equilibrium returns)

    Where Pi = equilibrium returns, P = pick matrix (views on assets),
    Q = view return vector, Omega = view uncertainty matrix,
    tau = scaling factor for uncertainty in equilibrium.

    Parameters (via kwargs to optimize)
    ------------------------------------
    P : np.ndarray, optional
        K x N pick matrix where K is the number of views and N is
        the number of assets. Each row identifies which assets a
        view refers to.
    Q : np.ndarray, optional
        K-length vector of expected returns for each view.
    omega : np.ndarray, optional
        K x K view uncertainty (covariance) matrix. If not provided,
        defaults to tau * diag(P @ Sigma @ P^T) (proportional to
        the uncertainty of each view).
    tau : float, optional
        Scalar controlling the uncertainty in the equilibrium prior.
        Default is 0.05.
    risk_aversion : float, optional
        Risk aversion coefficient (lambda). Default is 2.5.
    """

    name = "black_litterman"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values
        tau = kwargs.get("tau", self.params.get("tau", 0.05))
        risk_aversion = kwargs.get(
            "risk_aversion", self.params.get("risk_aversion", 2.5)
        )

        # Equilibrium weights (equal weight as proxy for market cap)
        w_mkt = np.ones(n_assets) / n_assets
        # Implied equilibrium returns
        pi = risk_aversion * cov @ w_mkt

        # Check for investor views
        P = kwargs.get("P", None)
        Q = kwargs.get("Q", None)

        if P is not None and Q is not None:
            P = np.asarray(P, dtype=float)
            Q = np.asarray(Q, dtype=float).ravel()

            # View uncertainty matrix
            omega = kwargs.get("omega", None)
            if omega is None:
                # Default: proportional to the variance of the view portfolios
                omega = np.diag(np.diag(tau * P @ cov @ P.T))
            else:
                omega = np.asarray(omega, dtype=float)

            # Posterior mean (Black-Litterman formula)
            tau_cov_inv = np.linalg.pinv(tau * cov)
            omega_inv = np.linalg.pinv(omega)

            # Posterior precision
            posterior_precision = tau_cov_inv + P.T @ omega_inv @ P
            posterior_cov = np.linalg.pinv(posterior_precision)

            # Posterior mean
            mu_bl = posterior_cov @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)
        else:
            # Without views, use equilibrium returns
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
