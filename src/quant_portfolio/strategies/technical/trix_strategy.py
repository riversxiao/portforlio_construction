"""TRIX Strategy - Triple EMA oscillator."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class TRIXStrategy(Strategy):
    """TRIX (triple exponential moving average) strategy.

    Uses the rate of change of triple-smoothed EMA for momentum.

    Parameters
    ----------
    period : int
        EMA period for each smoothing (default 15).
    signal_period : int
        Signal line period (default 9).
    """

    name = "trix_strategy"

    def __init__(self, period: int = 15, signal_period: int = 9, **kwargs):
        super().__init__(period=period, signal_period=signal_period, **kwargs)
        self.period = period
        self.signal_period = signal_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate TRIX signals."""
        close = data["close"]
        ema1 = close.ewm(span=self.period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        ema3 = ema2.ewm(span=self.period, adjust=False).mean()
        trix = ema3.pct_change() * 100
        signal_line = trix.ewm(span=self.signal_period, adjust=False).mean()

        signal = pd.Series(0, index=data.index)
        signal[trix > signal_line] = 1
        signal[trix < signal_line] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
