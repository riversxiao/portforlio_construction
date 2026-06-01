"""Time Series Decomposition algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class TimeSeriesDecompositionAlgorithm(Algorithm):
    """STL-like Time Series Decomposition for Allocation.

    Decomposes each asset's cumulative return series into trend,
    seasonal, and residual components using moving average approach.
    Allocates based on the trend component direction.
    """

    name = "time_series_decomposition"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        period = self.params.get("period", 21)

        trend_signals = np.zeros(n_assets)
        for i in range(n_assets):
            cum_ret = np.cumsum(ret_matrix[:, i])
            n = len(cum_ret)
            if n < period * 2:
                trend_signals[i] = cum_ret[-1] - cum_ret[0] if n > 0 else 0
                continue

            # Extract trend via centered moving average
            window = period
            pad = window // 2
            padded = np.pad(cum_ret, pad, mode='edge')
            trend = np.convolve(padded, np.ones(window) / window, mode='valid')
            if len(trend) > n:
                trend = trend[:n]

            # Trend slope at end
            trend_signals[i] = trend[-1] - trend[-min(period, len(trend))]

        scores = np.maximum(trend_signals, 0)
        total = scores.sum()
        if total > 1e-10:
            weights = scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
