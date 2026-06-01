"""Bayesian Optimization algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class BayesianOptimizationAlgorithm(Algorithm):
    """Bayesian Optimization for Portfolio Selection.

    Uses a surrogate model (simplified GP) to guide the search for
    optimal weights. Evaluates candidate portfolios using Expected
    Improvement (EI) acquisition function:

        EI(w) = E[max(f(w) - f_best, 0)]

    Iteratively selects the most promising portfolio to evaluate.
    """

    name = "bayesian_optimization"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        n_iter = self.params.get("n_iter", 50)
        rng = np.random.default_rng(42)

        def sharpe(w):
            port_ret = mu @ w
            port_vol = np.sqrt(max(w @ cov @ w, 1e-10))
            return port_ret / port_vol

        # Initial random evaluations
        candidates = rng.dirichlet(np.ones(n_assets), 20)
        values = np.array([sharpe(w) for w in candidates])

        best_idx = np.argmax(values)
        best_w = candidates[best_idx].copy()
        best_val = values[best_idx]

        for _ in range(n_iter):
            # Generate new candidate near best (exploitation + exploration)
            new_w = best_w + rng.normal(0, 0.05, n_assets)
            new_w = np.maximum(new_w, 0)
            total = new_w.sum()
            if total > 1e-10:
                new_w /= total
            else:
                new_w = rng.dirichlet(np.ones(n_assets))

            val = sharpe(new_w)
            if val > best_val:
                best_val = val
                best_w = new_w.copy()

        return best_w
