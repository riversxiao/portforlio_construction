"""Fourier Analysis algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class FourierAnalysisAlgorithm(Algorithm):
    """FFT Frequency Decomposition for Allocation.

    Applies Fast Fourier Transform to each asset's return series to
    identify dominant low-frequency components (trends). Filters out
    high-frequency noise and reconstructs the trend signal.

    Assets with stronger positive low-frequency components get higher
    weights.
    """

    name = "fourier_analysis"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        cutoff_pct = self.params.get("cutoff_pct", 0.1)

        trend_signals = np.zeros(n_assets)
        for i in range(n_assets):
            signal = ret_matrix[:, i]
            n = len(signal)
            if n < 4:
                trend_signals[i] = signal.mean()
                continue

            # FFT
            fft_vals = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(n)

            # Keep only low frequencies (below cutoff)
            cutoff_idx = max(1, int(len(freqs) * cutoff_pct))
            fft_filtered = np.zeros_like(fft_vals)
            fft_filtered[:cutoff_idx] = fft_vals[:cutoff_idx]

            # Reconstruct
            filtered = np.fft.irfft(fft_filtered, n=n)
            trend_signals[i] = filtered[-1]

        scores = np.maximum(trend_signals, 0)
        total = scores.sum()
        if total > 1e-10:
            weights = scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
