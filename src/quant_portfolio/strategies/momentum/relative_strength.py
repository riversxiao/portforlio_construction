"""Relative Strength Strategy - performance relative to a moving average benchmark."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class RelativeStrength(Strategy):
    """Relative strength strategy vs benchmark (moving average proxy).

    Compares the asset's performance against its own long-term moving average
    as a benchmark proxy.

    Parameters
    ----------
    lookback : int
        Lookback period for relative strength (default 50).
    ma_period : int
        Moving average period as benchmark (default 200).
    """

    name = "relative_strength"

    def __init__(self, lookback: int = 50, ma_period: int = 200, **kwargs):
        super().__init__(lookback=lookback, ma_period=ma_period, **kwargs)
        self.lookback = lookback
        self.ma_period = ma_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on relative strength vs moving average."""
        close = data["close"]
        ma = close.rolling(self.ma_period).mean()
        rs = close / (ma + 1e-10)
        rs_change = rs.pct_change(self.lookback)

        signal = pd.Series(0, index=data.index)
        signal[(close > ma) & (rs_change > 0)] = 1
        signal[(close < ma) & (rs_change < 0)] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
