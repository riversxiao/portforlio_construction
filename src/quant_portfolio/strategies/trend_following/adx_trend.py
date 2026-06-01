"""ADX Trend Strategy - Average Directional Index trend strength."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class ADXTrend(Strategy):
    """ADX-based trend strength strategy.

    Uses ADX to confirm trend strength and DI+/DI- for direction.

    Parameters
    ----------
    period : int
        ADX calculation period (default 14).
    adx_threshold : float
        Minimum ADX for trend confirmation (default 25).
    """

    name = "adx_trend"

    def __init__(self, period: int = 14, adx_threshold: float = 25, **kwargs):
        super().__init__(period=period, adx_threshold=adx_threshold, **kwargs)
        self.period = period
        self.adx_threshold = adx_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on ADX trend strength."""
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
        plus_di = 100 * (plus_dm.rolling(self.period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(self.period).mean() / (atr + 1e-10))

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.period).mean()

        signal = pd.Series(0, index=data.index)
        signal[(adx > self.adx_threshold) & (plus_di > minus_di)] = 1
        signal[(adx > self.adx_threshold) & (plus_di < minus_di)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
