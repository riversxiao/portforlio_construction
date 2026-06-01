"""Differential Evolution algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution as de_scipy

from quant_portfolio.algorithms.base import Algorithm


class DifferentialEvolutionAlgorithm(Algorithm):
    """Differential Evolution for Portfolio Optimization.

    Population-based stochastic optimizer using vector differences
    for perturbation. Well-suited for non-convex objectives.
    Uses scipy's differential_evolution with a portfolio objective
    that balances return and risk.
    """

    name = "differential_evolution"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values

        def objective(w):
            # Normalize to simplex
            w_norm = np.abs(w) / np.abs(w).sum()
            port_vol = np.sqrt(w_norm @ cov @ w_norm)
            port_ret = mu @ w_norm
            if port_vol < 1e-10:
                return 0.0
            return -port_ret / port_vol  # Negative Sharpe

        bounds = [(0, 1)] * n_assets
        result = de_scipy(objective, bounds, maxiter=100,
                          seed=42, tol=1e-6, polish=False)

        w = np.abs(result.x)
        w = w / w.sum() if w.sum() > 1e-10 else np.ones(n_assets) / n_assets
        return w
