"""Hilbert Transform algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class HilbertTransformAlgorithm(Algorithm):
    """Hilbert Transform for Instantaneous Phase Analysis.

    Applies the Hilbert transform to compute the analytic signal,
    extracting instantaneous amplitude and phase. Allocates based
    on instantaneous amplitude (signal strength):

        analytic = signal + j * H(signal)
        amplitude = |analytic|
    """

    name = "hilbert_transform"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values

        amplitudes = np.zeros(n_assets)
        for i in range(n_assets):
            signal = ret_matrix[:, i]
            n = len(signal)
            if n < 4:
                amplitudes[i] = abs(signal.mean())
                continue

            # Hilbert transform via FFT
            fft_vals = np.fft.fft(signal)
            h = np.zeros(n)
            if n % 2 == 0:
                h[0] = 1
                h[n // 2] = 1
                h[1:n // 2] = 2
            else:
                h[0] = 1
                h[1:(n + 1) // 2] = 2

            analytic = np.fft.ifft(fft_vals * h)
            # Instantaneous amplitude at the end
            amplitudes[i] = np.abs(analytic[-1])

        # Weight by amplitude (stronger signal = higher weight)
        total = amplitudes.sum()
        if total > 1e-10:
            weights = amplitudes / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
