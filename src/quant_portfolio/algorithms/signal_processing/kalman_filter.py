"""Kalman Filter Signal Extraction algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class KalmanFilterAlgorithm(Algorithm):
    """Kalman Filter for Trend Extraction and Allocation.

    Applies a univariate Kalman filter to each asset's cumulative return
    series to extract the underlying trend (state). Assets with stronger
    positive trend signals receive higher weights.

    State model: x_t = x_{t-1} + w_t  (random walk)
    Observation: y_t = x_t + v_t
    """

    name = "kalman_filter"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        q_var = self.params.get("process_noise", 0.001)
        r_var = self.params.get("observation_noise", 0.01)

        trend_signals = np.zeros(n_assets)
        for i in range(n_assets):
            # Kalman filter on cumulative returns
            obs = np.cumsum(ret_matrix[:, i])
            x_est = 0.0
            p_est = 1.0

            for y in obs:
                # Predict
                x_pred = x_est
                p_pred = p_est + q_var
                # Update
                k = p_pred / (p_pred + r_var)
                x_est = x_pred + k * (y - x_pred)
                p_est = (1 - k) * p_pred

            trend_signals[i] = x_est

        # Weight by positive trend strength
        scores = np.maximum(trend_signals, 0)
        total = scores.sum()
        if total > 1e-10:
            weights = scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
