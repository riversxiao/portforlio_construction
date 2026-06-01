"""Dynamic Correlation (DCC) algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class DynamicCorrelationAlgorithm(Algorithm):
    """DCC-GARCH Dynamic Correlation Allocation.

    Estimates time-varying correlations using exponentially weighted
    moving correlations (simplified DCC approach). Uses the most
    recent dynamic correlation matrix for minimum variance allocation.

    Q_t = (1-a-b)*Q_bar + a*e_{t-1}*e_{t-1}^T + b*Q_{t-1}
    """

    name = "dynamic_correlation"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        decay = self.params.get("decay", 0.94)

        # EWMA covariance
        ewm_cov = returns.ewm(span=int(1 / (1 - decay))).cov()
        # Get the last covariance matrix
        last_idx = returns.index[-1]
        cov_dynamic = ewm_cov.loc[last_idx].values

        if cov_dynamic.shape != (n_assets, n_assets):
            cov_dynamic = returns.cov().values

        # Minimum variance with dynamic covariance
        def objective(w):
            return w @ cov_dynamic @ w

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
