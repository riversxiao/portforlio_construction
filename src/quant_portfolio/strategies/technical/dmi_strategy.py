"""DMI Strategy - Directional Movement Index."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class DMIStrategy(Strategy):
    """Directional Movement Index strategy.

    Uses DI+ and DI- crossovers for trend direction signals.

    Parameters
    ----------
    period : int
        DMI calculation period (default 14).
    """

    name = "dmi_strategy"

    def __init__(self, period: int = 14, **kwargs):
        super().__init__(period=period, **kwargs)
        self.period = period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate DMI crossover signals."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0

        atr = tr.rolling(self.period).mean()
        plus_di = 100 * plus_dm.rolling(self.period).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(self.period).mean() / (atr + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[plus_di > minus_di] = 1
        signal[plus_di < minus_di] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
