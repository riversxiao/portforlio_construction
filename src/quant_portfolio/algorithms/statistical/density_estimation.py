"""Density Estimation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class DensityEstimationAlgorithm(Algorithm):
    """Kernel Density Estimation for Allocation.

    Estimates the probability density of each asset's returns using
    Gaussian KDE. Allocates based on the probability of positive
    returns (higher probability = higher weight):

        score_i = P(R_i > 0) estimated via KDE
    """

    name = "density_estimation"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values

        prob_positive = np.zeros(n_assets)
        for i in range(n_assets):
            series = ret_matrix[:, i]
            n = len(series)
            if n < 5:
                prob_positive[i] = 0.5
                continue

            # Gaussian KDE: P(X > 0)
            h = 1.06 * series.std() * n ** (-0.2)  # Silverman bandwidth
            if h < 1e-10:
                prob_positive[i] = 0.5
                continue

            # Evaluate CDF at 0 using KDE
            # P(X > 0) = 1 - mean(Phi((0 - x_i) / h))
            from scipy.stats import norm as norm_dist
            prob_positive[i] = 1.0 - norm_dist.cdf(-series / h).mean()

        scores = np.maximum(prob_positive, 0.01)
        weights = scores / scores.sum()
        return weights
