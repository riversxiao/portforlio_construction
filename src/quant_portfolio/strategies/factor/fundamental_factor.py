"""Fundamental Factor Strategy - composite fundamental score."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class FundamentalFactor(Strategy):
    """Composite fundamental factor strategy.

    Uses price and volume patterns as proxies for fundamental metrics
    (profitability, stability, valuation).

    Parameters
    ----------
    window : int
        Analysis window (default 60).
    """

    name = "fundamental_factor"

    def __init__(self, window: int = 60, **kwargs):
        super().__init__(window=window, **kwargs)
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate fundamental factor signals."""
        close = data["close"]
        volume = data["volume"]
        returns = close.pct_change()

        # Profitability proxy: consistent positive returns
        pos_ratio = (returns > 0).rolling(self.window).mean()

        # Stability proxy: low volatility
        vol = returns.rolling(self.window).std()
        vol_rank = vol.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Volume trend: increasing engagement
        vol_trend = volume.rolling(self.window).mean().pct_change(20)

        score = pos_ratio - vol_rank + 0.5 * np.sign(vol_trend)
        signal = pd.Series(0, index=data.index)
        signal[score > 0.5] = 1
        signal[score < -0.5] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
