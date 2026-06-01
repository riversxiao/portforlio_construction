"""Parabolic SAR Strategy - Stop and Reverse trailing stop."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class ParabolicSAR(Strategy):
    """Parabolic SAR trend following strategy.

    Uses the Parabolic Stop and Reverse indicator for trend direction.

    Parameters
    ----------
    af_start : float
        Initial acceleration factor (default 0.02).
    af_max : float
        Maximum acceleration factor (default 0.2).
    af_step : float
        Acceleration factor increment (default 0.02).
    """

    name = "parabolic_sar"

    def __init__(self, af_start: float = 0.02, af_max: float = 0.2,
                 af_step: float = 0.02, **kwargs):
        super().__init__(af_start=af_start, af_max=af_max, af_step=af_step, **kwargs)
        self.af_start = af_start
        self.af_max = af_max
        self.af_step = af_step

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Parabolic SAR signals."""
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values
        n = len(data)

        sar = np.zeros(n)
        direction = np.ones(n)
        af = self.af_start
        ep = high[0]
        sar[0] = low[0]

        for i in range(1, n):
            if direction[i - 1] == 1:
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
                sar[i] = min(sar[i], low[i - 1])
                if low[i] < sar[i]:
                    direction[i] = -1
                    sar[i] = ep
                    ep = low[i]
                    af = self.af_start
                else:
                    direction[i] = 1
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + self.af_step, self.af_max)
            else:
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
                sar[i] = max(sar[i], high[i - 1])
                if high[i] > sar[i]:
                    direction[i] = 1
                    sar[i] = ep
                    ep = high[i]
                    af = self.af_start
                else:
                    direction[i] = -1
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + self.af_step, self.af_max)

        signal = pd.Series(direction.astype(int), index=data.index)
        signal[signal == 0] = 1
        return pd.DataFrame({"signal": signal}, index=data.index)
