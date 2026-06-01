"""Copula Modeling algorithm."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from quant_portfolio.algorithms.base import Algorithm


class CopulaModelingAlgorithm(Algorithm):
    """Copula-Based Dependence Modeling for Allocation.

    Models the dependence structure between assets using a Gaussian
    copula. Transforms returns to uniform marginals via empirical CDF,
    then estimates the copula correlation matrix. Uses this for
    diversification-based allocation.

    Assets with lower copula correlations to the rest receive higher
    weights (better diversifiers).
    """

    name = "copula_modeling"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_obs = len(ret_matrix)

        # Transform to uniform marginals (empirical CDF)
        U = np.zeros_like(ret_matrix)
        for i in range(n_assets):
            ranks = ret_matrix[:, i].argsort().argsort()
            U[:, i] = (ranks + 1) / (n_obs + 1)

        # Transform to standard normal (Gaussian copula)
        Z = norm.ppf(np.clip(U, 0.001, 0.999))

        # Copula correlation matrix
        copula_corr = np.corrcoef(Z, rowvar=False)

        # Allocate inversely to average copula correlation
        avg_corr = np.abs(copula_corr).mean(axis=1)
        inv_corr = 1.0 / np.maximum(avg_corr, 1e-10)
        weights = inv_corr / inv_corr.sum()
        return weights
