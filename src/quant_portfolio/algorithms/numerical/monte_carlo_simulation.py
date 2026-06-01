"""Monte Carlo Simulation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class MonteCarloSimulationAlgorithm(Algorithm):
    """Monte Carlo Simulation for Portfolio Selection.

    Generates random portfolios on the efficient frontier by
    sampling weight vectors uniformly from the simplex.
    Selects the portfolio with the highest Sharpe ratio from
    the simulated set.
    """

    name = "monte_carlo_simulation"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        n_portfolios = self.params.get("n_portfolios", 5000)
        rng = np.random.default_rng(42)

        # Generate random portfolios from Dirichlet
        weights_array = rng.dirichlet(np.ones(n_assets), n_portfolios)

        best_sharpe = -np.inf
        best_w = np.ones(n_assets) / n_assets

        for i in range(n_portfolios):
            w = weights_array[i]
            port_ret = mu @ w
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol > 1e-10:
                sharpe = port_ret / port_vol
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_w = w

        return best_w
