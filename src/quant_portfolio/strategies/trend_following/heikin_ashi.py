"""Heikin-Ashi Strategy - Heikin-Ashi candle trend."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class HeikinAshi(Strategy):
    """Heikin-Ashi candle-based trend strategy.

    Generates signals based on consecutive Heikin-Ashi candle colors.

    Parameters
    ----------
    consecutive : int
        Number of consecutive candles for confirmation (default 3).
    """

    name = "heikin_ashi"

    def __init__(self, consecutive: int = 3, **kwargs):
        super().__init__(consecutive=consecutive, **kwargs)
        self.consecutive = consecutive

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Heikin-Ashi trend signals."""
        o = data["open"].values
        h = data["high"].values
        l = data["low"].values
        c = data["close"].values
        n = len(data)

        ha_close = (o + h + l + c) / 4
        ha_open = np.zeros(n)
        ha_open[0] = (o[0] + c[0]) / 2

        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2

        bullish = ha_close > ha_open

        signal = pd.Series(0, index=data.index)
        for i in range(self.consecutive, n):
            if all(bullish[i - self.consecutive + 1:i + 1]):
                signal.iloc[i] = 1
            elif all(~bullish[i - self.consecutive + 1:i + 1]):
                signal.iloc[i] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
