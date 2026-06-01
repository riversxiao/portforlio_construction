"""Marginal Risk Contribution algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class MarginalRiskAlgorithm(Algorithm):
    """Marginal Risk Contribution Weighting.

    Allocates inversely proportional to marginal risk contribution.
    The marginal risk of asset i is:

        MRC_i = (Sigma * w)_i / sigma_p

    Assets with higher marginal risk get lower weights, producing
    a portfolio where adding to any position has similar risk impact.
    """

    name = "marginal_risk"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values

        # Start with equal weights to compute marginal risks
        w = np.ones(n_assets) / n_assets
        # Iterative refinement
        for _ in range(20):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                break
            mrc = (cov @ w) / port_vol
            mrc = np.maximum(mrc, 1e-10)
            w_new = (1.0 / mrc) / (1.0 / mrc).sum()
            if np.allclose(w, w_new, atol=1e-8):
                break
            w = w_new

        return w
