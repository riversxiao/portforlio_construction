"""PCA Strategy - dimensionality reduction for signal extraction."""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from quant_portfolio.strategies.base import Strategy


class PCAStrategy(Strategy):
    """PCA dimensionality reduction strategy.

    Extracts principal components from technical features and uses
    the first component as a trading signal.

    Parameters
    ----------
    window : int
        Rolling window for feature construction (default 60).
    n_components : int
        Number of PCA components (default 2).
    """

    name = "pca_strategy"

    def __init__(self, window: int = 60, n_components: int = 2, **kwargs):
        super().__init__(window=window, n_components=n_components, **kwargs)
        self.window = window
        self.n_components = n_components

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using PCA feature extraction."""
        close = data["close"]
        returns = close.pct_change()

        features = pd.DataFrame(index=data.index)
        for lag in [1, 2, 5, 10, 20]:
            features[f"ret_{lag}"] = close.pct_change(lag)
        features["vol"] = returns.rolling(10).std()
        features = features.dropna()

        if len(features) < self.window:
            return pd.DataFrame({"signal": 0}, index=data.index)

        pca = PCA(n_components=self.n_components, random_state=42)
        X = features.values
        components = pca.fit_transform(X)

        # Use first component as signal
        pc1 = pd.Series(components[:, 0], index=features.index)
        pc1_z = (pc1 - pc1.rolling(self.window).mean()) / (pc1.rolling(self.window).std() + 1e-10)

        signal = pd.Series(0, index=data.index)
        for idx in features.index:
            z_val = pc1_z.get(idx, 0)
            if pd.notna(z_val):
                if z_val > 1.0:
                    signal[idx] = 1
                elif z_val < -1.0:
                    signal[idx] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
