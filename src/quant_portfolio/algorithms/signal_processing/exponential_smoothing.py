"""Exponential Smoothing (Holt-Winters) algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class ExponentialSmoothingAlgorithm(Algorithm):
    """Holt Double Exponential Smoothing for Trend Allocation.

    Applies Holt's linear trend method to extract level and trend:

        level_t = alpha * y_t + (1-alpha) * (level_{t-1} + trend_{t-1})
        trend_t = beta * (level_t - level_{t-1}) + (1-beta) * trend_{t-1}

    Assets with positive trend estimates receive higher weights.
    """

    name = "exponential_smoothing"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        alpha = self.params.get("alpha", 0.3)
        beta = self.params.get("beta", 0.1)

        trend_signals = np.zeros(n_assets)
        for i in range(n_assets):
            cum_ret = np.cumsum(ret_matrix[:, i])
            n = len(cum_ret)
            if n < 2:
                trend_signals[i] = 0
                continue

            # Initialize
            level = cum_ret[0]
            trend = cum_ret[1] - cum_ret[0] if n > 1 else 0

            for t in range(1, n):
                new_level = alpha * cum_ret[t] + (1 - alpha) * (level + trend)
                new_trend = beta * (new_level - level) + (1 - beta) * trend
                level = new_level
                trend = new_trend

            trend_signals[i] = trend

        scores = np.maximum(trend_signals, 0)
        total = scores.sum()
        if total > 1e-10:
            weights = scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
