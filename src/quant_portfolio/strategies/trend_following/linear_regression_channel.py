"""Linear Regression Channel Strategy - regression channel breakout."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class LinearRegressionChannel(Strategy):
    """Linear regression channel trend strategy.

    Fits a rolling linear regression and trades breakouts above/below
    the channel boundaries.

    Parameters
    ----------
    window : int
        Rolling window for regression (default 50).
    num_std : float
        Number of standard deviations for channel (default 2.0).
    """

    name = "linear_regression_channel"

    def __init__(self, window: int = 50, num_std: float = 2.0, **kwargs):
        super().__init__(window=window, num_std=num_std, **kwargs)
        self.window = window
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on regression channel."""
        close = data["close"]
        signal = pd.Series(0, index=data.index)

        for i in range(self.window, len(data)):
            y = close.iloc[i - self.window:i].values
            x = np.arange(self.window)
            slope, intercept = np.polyfit(x, y, 1)
            fitted = slope * x + intercept
            residuals = y - fitted
            std = residuals.std()

            projected = slope * self.window + intercept
            current = close.iloc[i]
            deviation = (current - projected) / (std + 1e-10)

            if slope > 0 and deviation > -self.num_std:
                signal.iloc[i] = 1
            elif slope < 0 and deviation < self.num_std:
                signal.iloc[i] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
