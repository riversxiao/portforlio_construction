"""Kalman Filter Mean Reversion Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class KalmanFilterMR(Strategy):
    """Kalman filter-based mean reversion strategy.

    Uses a simple Kalman filter to estimate the fair value and trades
    deviations from the filtered estimate.

    Parameters
    ----------
    transition_covariance : float
        Process noise (default 0.01).
    observation_covariance : float
        Observation noise (default 1.0).
    entry_threshold : float
        Entry threshold as fraction of filtered value (default 0.02).
    """

    name = "kalman_filter_mr"

    def __init__(self, transition_covariance: float = 0.01,
                 observation_covariance: float = 1.0,
                 entry_threshold: float = 0.02, **kwargs):
        super().__init__(transition_covariance=transition_covariance,
                         observation_covariance=observation_covariance,
                         entry_threshold=entry_threshold, **kwargs)
        self.transition_cov = transition_covariance
        self.observation_cov = observation_covariance
        self.entry_threshold = entry_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using Kalman filter estimate."""
        close = data["close"].values
        n = len(close)

        # Simple 1D Kalman filter
        x_est = close[0]
        p_est = 1.0
        filtered = np.zeros(n)

        for i in range(n):
            # Predict
            x_pred = x_est
            p_pred = p_est + self.transition_cov

            # Update
            k_gain = p_pred / (p_pred + self.observation_cov)
            x_est = x_pred + k_gain * (close[i] - x_pred)
            p_est = (1 - k_gain) * p_pred
            filtered[i] = x_est

        deviation = (close - filtered) / (filtered + 1e-10)
        signal = np.zeros(n)
        signal[deviation < -self.entry_threshold] = 1
        signal[deviation > self.entry_threshold] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
