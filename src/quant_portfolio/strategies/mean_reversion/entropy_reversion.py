"""Entropy Reversion Strategy - information entropy based."""

import pandas as pd
import numpy as np

from quant_portfolio.strategies.base import Strategy


class EntropyReversion(Strategy):
    """Entropy-based mean reversion strategy.

    Uses Shannon entropy of binned returns to detect low-entropy
    (predictable/mean-reverting) periods for trading.

    Parameters
    ----------
    window : int
        Window for entropy calculation (default 30).
    n_bins : int
        Number of bins for entropy estimation (default 10).
    entry_z : float
        Z-score threshold for entry (default 1.5).
    """

    name = "entropy_reversion"

    def __init__(self, window: int = 30, n_bins: int = 10,
                 entry_z: float = 1.5, **kwargs):
        super().__init__(window=window, n_bins=n_bins, entry_z=entry_z, **kwargs)
        self.window = window
        self.n_bins = n_bins
        self.entry_z = entry_z

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on entropy regime detection."""
        close = data["close"]
        returns = close.pct_change()
        signal = pd.Series(0, index=data.index)

        for i in range(self.window, len(data)):
            r = returns.iloc[i - self.window:i].dropna().values
            if len(r) < 10:
                continue
            counts, _ = np.histogram(r, bins=self.n_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(self.n_bins)

            if entropy < 0.5 * max_entropy:
                mu = close.iloc[i - self.window:i].mean()
                std = close.iloc[i - self.window:i].std()
                if std < 1e-10:
                    continue
                z = (close.iloc[i] - mu) / std
                if z < -self.entry_z:
                    signal.iloc[i] = 1
                elif z > self.entry_z:
                    signal.iloc[i] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
