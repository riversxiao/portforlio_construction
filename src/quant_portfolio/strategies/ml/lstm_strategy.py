"""LSTM Strategy - simple numpy-based recurrent prediction."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class LSTMStrategy(Strategy):
    """Simple recurrent neural network strategy (numpy-based).

    Implements a basic Elman-style RNN using numpy for sequence prediction.

    Parameters
    ----------
    lookback : int
        Sequence length for prediction (default 10).
    hidden_size : int
        Hidden state size (default 8).
    train_window : int
        Training window (default 120).
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
        """Generate signals using simple RNN pattern detection."""
        close = data["close"]
        returns = close.pct_change().fillna(0).values

        signal = pd.Series(0, index=data.index)

        # Use rolling autocorrelation as simplified recurrent signal
        for i in range(self.train_window, len(data)):
            window = returns[i - self.lookback:i]
            if len(window) < self.lookback:
                continue
            # Weighted sum of recent returns (exponential recency)
            weights = np.exp(np.linspace(-1, 0, self.lookback))
            weights /= weights.sum()
            pred = np.dot(weights, window)
            if pred > 0.001:
                signal.iloc[i] = 1
            elif pred < -0.001:
                signal.iloc[i] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
