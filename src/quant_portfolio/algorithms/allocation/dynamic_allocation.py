"""Dynamic Allocation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class DynamicAllocationAlgorithm(Algorithm):
    """Dynamic Asset Allocation.

    Time-varying allocation that adjusts based on changing market
    conditions. Uses exponentially weighted moments to capture
    recent dynamics:

        mu_t = EWMA(returns, span)
        Sigma_t = EWMA(covariance, span)

    Then applies inverse-variance weighting with dynamic estimates.
    """

    name = "dynamic_allocation"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        span = self.params.get("span", 60)

        # Exponentially weighted variance
        ewm_var = returns.ewm(span=span).var().iloc[-1].values
        ewm_var = np.maximum(ewm_var, 1e-10)

        # Inverse variance with EWM estimates
        inv_var = 1.0 / ewm_var
        weights = inv_var / inv_var.sum()
        return weights
