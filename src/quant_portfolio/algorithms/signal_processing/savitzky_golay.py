"""Savitzky-Golay Smoothing algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class SavitzkyGolayAlgorithm(Algorithm):
    """Savitzky-Golay Filter for Trend Smoothing.

    Applies local polynomial regression (Savitzky-Golay filter)
    to smooth return series while preserving higher-order moments.
    Allocates based on the smoothed trend direction.

    Uses a polynomial of degree p fitted over a window of 2m+1 points.
    """

    name = "savitzky_golay"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        window = self.params.get("window", 11)
        poly_order = self.params.get("poly_order", 3)

        # Ensure window is odd
        if window % 2 == 0:
            window += 1
        window = min(window, len(ret_matrix))
        if window <= poly_order:
            window = poly_order + 2
            if window % 2 == 0:
                window += 1

        trend_signals = np.zeros(n_assets)
        half_w = window // 2

        for i in range(n_assets):
            cum_ret = np.cumsum(ret_matrix[:, i])
            n = len(cum_ret)
            if n < window:
                trend_signals[i] = cum_ret[-1] if n > 0 else 0
                continue

            # Compute SG coefficients for smoothing
            # Fit polynomial to last window points
            x = np.arange(window) - half_w
            last_window = cum_ret[-window:]
            coeffs = np.polyfit(x, last_window, poly_order)
            # Evaluate at center (x=0 gives smoothed value)
            smoothed_last = np.polyval(coeffs, half_w)
            smoothed_mid = np.polyval(coeffs, 0)
            trend_signals[i] = smoothed_last - smoothed_mid

        scores = np.maximum(trend_signals, 0)
        total = scores.sum()
        if total > 1e-10:
            weights = scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
