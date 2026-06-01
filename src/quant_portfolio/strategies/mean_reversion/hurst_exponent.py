"""Hurst Exponent Strategy - regime detection for mean reversion."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class HurstExponent(Strategy):
    """Hurst exponent regime detection strategy.

    Estimates the Hurst exponent to determine if the series is
    mean-reverting (H < 0.5) and trades accordingly.

    Parameters
    ----------
    window : int
        Window for Hurst exponent estimation (default 100).
    entry_z : float
        Z-score entry threshold when mean-reverting (default 1.5).
    """

    name = "hurst_exponent"

    def __init__(self, window: int = 100, entry_z: float = 1.5, **kwargs):
        super().__init__(window=window, entry_z=entry_z, **kwargs)
        self.window = window
        self.entry_z = entry_z

    def _estimate_hurst(self, ts: np.ndarray) -> float:
        """Estimate Hurst exponent using R/S analysis."""
        n = len(ts)
        if n < 20:
            return 0.5
        max_k = min(n // 2, 50)
        lags = range(2, max_k)
        tau = []
        for lag in lags:
            segments = n // lag
            if segments < 1:
                break
            rs_values = []
            for seg in range(segments):
                segment = ts[seg * lag:(seg + 1) * lag]
                mean_seg = segment.mean()
                deviations = segment - mean_seg
                cumdev = np.cumsum(deviations)
                r = cumdev.max() - cumdev.min()
                s = segment.std()
                if s > 1e-10:
                    rs_values.append(r / s)
            if rs_values:
                tau.append(np.mean(rs_values))
            else:
                tau.append(1.0)

        if len(tau) < 3:
            return 0.5
        log_lags = np.log(list(lags)[:len(tau)])
        log_tau = np.log(np.array(tau) + 1e-10)
        coeffs = np.polyfit(log_lags, log_tau, 1)
        return coeffs[0]

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on Hurst exponent regime."""
        close = data["close"]
        log_price = np.log(close.values)

        signal = pd.Series(0, index=data.index)
        for i in range(self.window, len(data)):
            window_data = log_price[i - self.window:i + 1]
            h = self._estimate_hurst(window_data)
            if h < 0.4:  # Mean-reverting regime
                mu = window_data.mean()
                std = window_data.std()
                if std < 1e-10:
                    continue
                z = (window_data[-1] - mu) / std
                if z < -self.entry_z:
                    signal.iloc[i] = 1
                elif z > self.entry_z:
                    signal.iloc[i] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
