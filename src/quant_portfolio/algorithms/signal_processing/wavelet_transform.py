"""Wavelet Transform Denoising algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class WaveletTransformAlgorithm(Algorithm):
    """Wavelet Transform Denoising for Signal-Based Allocation.

    Applies a simple Haar wavelet decomposition to denoise return
    series, then uses the denoised trend for allocation. The Haar
    wavelet computes pairwise averages and differences:

        approximation = (x[2i] + x[2i+1]) / 2
        detail = (x[2i] - x[2i+1]) / 2

    Reconstructs using only approximation coefficients (low-frequency).
    """

    name = "wavelet_transform"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_levels = self.params.get("n_levels", 3)

        trend_signals = np.zeros(n_assets)
        for i in range(n_assets):
            signal = ret_matrix[:, i].copy()
            # Haar wavelet decomposition - keep only approximation
            approx = signal.copy()
            for _ in range(n_levels):
                n = len(approx)
                if n < 2:
                    break
                n_even = n - n % 2
                approx = (approx[:n_even:2] + approx[1:n_even:2]) / 2.0

            # Trend signal is the mean of approximation
            trend_signals[i] = approx.mean() if len(approx) > 0 else 0.0

        scores = np.maximum(trend_signals, 0)
        total = scores.sum()
        if total > 1e-10:
            weights = scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
