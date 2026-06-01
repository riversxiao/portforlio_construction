"""Bandpass Filter algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class BandpassFilterAlgorithm(Algorithm):
    """Bandpass Filter for Cycle Extraction.

    Applies a frequency-domain bandpass filter to isolate signals
    within a specific frequency band (e.g., business cycle frequencies).
    Allocates based on the phase and amplitude of the filtered signal.

    Uses FFT to filter, keeping only frequencies within [f_low, f_high].
    """

    name = "bandpass_filter"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        # Period bounds in days (e.g., 20-60 day cycles)
        low_period = self.params.get("low_period", 20)
        high_period = self.params.get("high_period", 60)

        signals = np.zeros(n_assets)
        for i in range(n_assets):
            series = ret_matrix[:, i]
            n = len(series)
            if n < high_period:
                signals[i] = series.mean()
                continue

            fft_vals = np.fft.rfft(series)
            freqs = np.fft.rfftfreq(n)

            # Convert period bounds to frequency bounds
            f_low = 1.0 / high_period
            f_high = 1.0 / low_period

            # Bandpass: zero out frequencies outside band
            mask = (freqs >= f_low) & (freqs <= f_high)
            filtered_fft = np.zeros_like(fft_vals)
            filtered_fft[mask] = fft_vals[mask]

            filtered = np.fft.irfft(filtered_fft, n=n)
            signals[i] = filtered[-1]

        scores = np.maximum(signals, 0)
        total = scores.sum()
        if total > 1e-10:
            weights = scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
