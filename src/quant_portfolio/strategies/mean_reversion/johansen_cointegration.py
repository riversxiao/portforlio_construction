"""Johansen Cointegration Strategy - multivariate cointegration."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class JohansenCointegration(Strategy):
    """Johansen cointegration-based mean reversion.

    Constructs a mean-reverting spread using the ratio of price to its
    exponential moving average as a proxy for cointegration residual.

    Parameters
    ----------
    window : int
        Window for spread calculation (default 60).
    ema_span : int
        EMA span for equilibrium estimate (default 120).
    entry_z : float
        Z-score entry threshold (default 2.0).
    """

    name = "johansen_cointegration"

    def __init__(self, window: int = 60, ema_span: int = 120,
                 entry_z: float = 2.0, **kwargs):
        super().__init__(window=window, ema_span=ema_span, entry_z=entry_z, **kwargs)
        self.window = window
        self.ema_span = ema_span
        self.entry_z = entry_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate cointegration-based mean reversion signals."""
        close = data["close"]
        ema = close.ewm(span=self.ema_span, adjust=False).mean()
        spread = np.log(close) - np.log(ema)
        spread_mean = spread.rolling(self.window).mean()
        spread_std = spread.rolling(self.window).std()
        zscore = (spread - spread_mean) / (spread_std + 1e-10)

        signal = pd.Series(0, index=data.index)
        signal[zscore < -self.entry_z] = 1
        signal[zscore > self.entry_z] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
