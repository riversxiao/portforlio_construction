"""Changepoint Detection algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class ChangepointDetectionAlgorithm(Algorithm):
    """Structural Break / Changepoint Detection Allocation.

    Detects structural breaks in asset return distributions using
    CUSUM (Cumulative Sum) statistics. Assets with recent structural
    breaks receive lower weights (higher uncertainty), while stable
    assets receive higher weights.
    """

    name = "changepoint_detection"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        threshold = self.params.get("threshold", 2.0)

        stability_scores = np.zeros(n_assets)
        for i in range(n_assets):
            series = ret_matrix[:, i]
            n = len(series)
            if n < 10:
                stability_scores[i] = 1.0
                continue

            # CUSUM statistic
            mean_val = series.mean()
            cusum = np.cumsum(series - mean_val)
            # Max absolute CUSUM normalized by sqrt(n) and std
            std_val = max(series.std(), 1e-10)
            max_cusum = np.abs(cusum).max() / (std_val * np.sqrt(n))

            # Higher CUSUM = more likely changepoint = less stable
            if max_cusum > threshold:
                stability_scores[i] = 1.0 / (max_cusum + 1)
            else:
                stability_scores[i] = 1.0

        total = stability_scores.sum()
        if total > 1e-10:
            weights = stability_scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
