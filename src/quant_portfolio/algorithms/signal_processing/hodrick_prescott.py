"""Hodrick-Prescott Filter algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class HodrickPrescottAlgorithm(Algorithm):
    """Hodrick-Prescott Filter for Trend Extraction.

    Decomposes time series into trend and cycle components by solving:

        min sum((y_t - tau_t)^2) + lambda * sum((tau_{t+1} - 2*tau_t + tau_{t-1})^2)

    where lambda controls smoothness. Uses the extracted trend
    direction to allocate weights.
    """

    name = "hodrick_prescott"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        lam = self.params.get("lambda_hp", 1600)

        trend_signals = np.zeros(n_assets)
        for i in range(n_assets):
            cum_ret = np.cumsum(ret_matrix[:, i])
            n = len(cum_ret)
            if n < 4:
                trend_signals[i] = cum_ret[-1] if n > 0 else 0
                continue

            # Construct second difference matrix
            D = np.zeros((n - 2, n))
            for j in range(n - 2):
                D[j, j] = 1
                D[j, j + 1] = -2
                D[j, j + 2] = 1

            # Solve: (I + lambda * D^T D) * tau = y
            A = np.eye(n) + lam * (D.T @ D)
            trend = np.linalg.solve(A, cum_ret)
            # Trend direction: last value minus mean
            trend_signals[i] = trend[-1] - trend[max(0, n // 2)]

        scores = np.maximum(trend_signals, 0)
        total = scores.sum()
        if total > 1e-10:
            weights = scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
