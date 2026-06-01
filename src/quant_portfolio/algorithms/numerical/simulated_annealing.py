"""Simulated Annealing algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class SimulatedAnnealingAlgorithm(Algorithm):
    """Simulated Annealing for Portfolio Optimization.

    Metaheuristic that explores the weight space by accepting worse
    solutions with decreasing probability (temperature schedule).
    Minimizes portfolio variance with Sharpe ratio bonus:

        E = w^T*Sigma*w - lambda * mu^T*w

    Temperature decreases geometrically: T_{k+1} = alpha * T_k
    """

    name = "simulated_annealing"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        n_iter = self.params.get("n_iter", 1000)
        temp_init = self.params.get("temp_init", 1.0)
        cooling = self.params.get("cooling", 0.995)
        rng = np.random.default_rng(42)

        def energy(w):
            return w @ cov @ w - 0.5 * mu @ w

        w = np.ones(n_assets) / n_assets
        best_w = w.copy()
        best_e = energy(w)
        temp = temp_init

        for _ in range(n_iter):
            # Perturb weights
            perturbation = rng.normal(0, 0.05, n_assets)
            w_new = w + perturbation
            w_new = np.maximum(w_new, 0)
            w_new = w_new / w_new.sum()

            e_new = energy(w_new)
            delta = e_new - energy(w)

            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
                w = w_new
                if e_new < best_e:
                    best_w = w_new.copy()
                    best_e = e_new

            temp *= cooling

        return best_w
