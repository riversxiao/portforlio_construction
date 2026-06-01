"""CCI Strategy - Commodity Channel Index."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class CCIStrategy(Strategy):
    """CCI oscillator strategy.

    Uses Commodity Channel Index for overbought/oversold signals.

    Parameters
    ----------
    period : int
        CCI calculation period (default 20).
    upper : float
        Overbought threshold (default 100).
    lower : float
        Oversold threshold (default -100).
    """

    name = "cci_strategy"

    def __init__(self, period: int = 20, upper: float = 100,
                 lower: float = -100, **kwargs):
        super().__init__(period=period, upper=upper, lower=lower, **kwargs)
        self.period = period
        self.upper = upper
        self.lower = lower

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate CCI signals."""
        tp = (data["high"] + data["low"] + data["close"]) / 3
        ma = tp.rolling(self.period).mean()
        mad = tp.rolling(self.period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        cci = (tp - ma) / (0.015 * mad + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[cci < self.lower] = 1
        signal[cci > self.upper] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
