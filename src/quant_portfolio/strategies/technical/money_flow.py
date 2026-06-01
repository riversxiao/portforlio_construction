"""Money Flow Index Strategy."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class MoneyFlow(Strategy):
    """Money Flow Index (MFI) strategy.

    Volume-weighted RSI that identifies overbought/oversold conditions.

    Parameters
    ----------
    period : int
        MFI calculation period (default 14).
    overbought : float
        Overbought threshold (default 80).
    oversold : float
        Oversold threshold (default 20).
    """

    name = "money_flow"

    def __init__(self, period: int = 14, overbought: float = 80,
                 oversold: float = 20, **kwargs):
        super().__init__(period=period, overbought=overbought,
                         oversold=oversold, **kwargs)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MFI signals."""
        tp = (data["high"] + data["low"] + data["close"]) / 3
        mf = tp * data["volume"]
        tp_diff = tp.diff()

        pos_mf = mf.where(tp_diff > 0, 0).rolling(self.period).sum()
        neg_mf = mf.where(tp_diff < 0, 0).rolling(self.period).sum()
        mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))

        signal = pd.Series(0, index=data.index)
        signal[mfi < self.oversold] = 1
        signal[mfi > self.overbought] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
