"""Recurrent Weighted Strategy - EMA-inspired sequential signal generation."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class RecurrentWeightedStrategy(Strategy):
    """Exponential-weighted sequential signal strategy.

    Applies an exponential recency weighting to recent returns to generate
    directional signals. This is inspired by LSTM gating mechanisms but
    uses a simplified, parameter-free exponential weighting scheme rather
    than a trained recurrent neural network.

    The approach approximates the "forget gate" behavior of an LSTM by
    weighting recent observations exponentially, giving more influence
    to the most recent data points.

    Parameters
    ----------
    lookback : int
        Sequence length for the weighted sum (default 10).
    hidden_size : int
        Unused, retained for API compatibility (default 8).
    train_window : int
        Minimum observations before generating signals (default 120).
    """

    name = "lstm_strategy"

    def __init__(self, lookback: int = 10, hidden_size: int = 8,
                 train_window: int = 120, **kwargs):
        super().__init__(lookback=lookback, hidden_size=hidden_size,
                         train_window=train_window, **kwargs)
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.train_window = train_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using exponential-weighted recent returns.

        Computes a weighted average of the last `lookback` returns using
        exponentially increasing weights (more weight on recent returns).
        Generates a long signal when the weighted average exceeds +0.001
        and a short signal when it falls below -0.001.
        """
        close = data["close"]
        returns = close.pct_change().fillna(0).values

        signal = pd.Series(0, index=data.index)

        # Exponential weight vector (recency-biased)
        weights = np.exp(np.linspace(-1, 0, self.lookback))
        weights /= weights.sum()

        for i in range(self.train_window, len(data)):
            window = returns[i - self.lookback:i]
            if len(window) < self.lookback:
                continue
            pred = np.dot(weights, window)
            if pred > 0.001:
                signal.iloc[i] = 1
            elif pred < -0.001:
                signal.iloc[i] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)


# Backward-compatible alias
LSTMStrategy = RecurrentWeightedStrategy
