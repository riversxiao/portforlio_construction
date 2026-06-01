"""Multi-Factor Strategy - combined multi-factor model."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class MultiFactor(Strategy):
    """Combined multi-factor strategy.

    Combines value, momentum, and quality factors into a composite signal.

    Parameters
    ----------
    momentum_window : int
        Momentum lookback (default 60).
    value_window : int
        Value lookback (default 252).
    quality_window : int
        Quality (Sharpe) lookback (default 120).
    """

    name = "multi_factor"

    def __init__(self, momentum_window: int = 60, value_window: int = 252,
                 quality_window: int = 120, **kwargs):
        super().__init__(momentum_window=momentum_window, value_window=value_window,
                         quality_window=quality_window, **kwargs)
        self.momentum_window = momentum_window
        self.value_window = value_window
        self.quality_window = quality_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate multi-factor composite signals."""
        close = data["close"]
        returns = close.pct_change()

        # Momentum score
        mom = close.pct_change(self.momentum_window)
        mom_score = np.sign(mom)

        # Value score
        long_ma = close.rolling(self.value_window).mean()
        value_ratio = close / (long_ma + 1e-10)
        val_score = -np.sign(value_ratio - 1)

        # Quality score
        sharpe = returns.rolling(self.quality_window).mean() / (
            returns.rolling(self.quality_window).std() + 1e-10
        )
        qual_score = np.sign(sharpe)

        composite = (mom_score + val_score + qual_score) / 3.0
        signal = pd.Series(0, index=data.index)
        signal[composite > 0.3] = 1
        signal[composite < -0.3] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
