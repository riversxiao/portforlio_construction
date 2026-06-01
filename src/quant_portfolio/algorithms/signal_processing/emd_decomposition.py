"""Empirical Mode Decomposition algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class EMDDecompositionAlgorithm(Algorithm):
    """Empirical Mode Decomposition (EMD) for Signal Analysis.

    Decomposes signals into Intrinsic Mode Functions (IMFs) using a
    simplified sifting process. The residual (trend) after removing
    oscillatory components is used for allocation.

    Simplified EMD: iteratively subtract local mean to extract
    oscillatory modes, leaving the trend as residual.
    """

    name = "emd_decomposition"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_imfs = self.params.get("n_imfs", 3)

        trend_signals = np.zeros(n_assets)
        for i in range(n_assets):
            signal = np.cumsum(ret_matrix[:, i])
            residual = signal.copy()

            for _ in range(n_imfs):
                n = len(residual)
                if n < 4:
                    break
                # Simple local mean estimation using moving average
                window = max(3, n // 10)
                if window % 2 == 0:
                    window += 1
                pad = window // 2
                padded = np.pad(residual, pad, mode='edge')
                local_mean = np.convolve(padded, np.ones(window) / window, mode='valid')
                if len(local_mean) > n:
                    local_mean = local_mean[:n]
                elif len(local_mean) < n:
                    local_mean = np.pad(local_mean, (0, n - len(local_mean)), mode='edge')
                imf = residual - local_mean
                residual = local_mean

            # Trend signal from residual
            trend_signals[i] = residual[-1] - residual[len(residual) // 2]

        scores = np.maximum(trend_signals, 0)
        total = scores.sum()
        if total > 1e-10:
            weights = scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
