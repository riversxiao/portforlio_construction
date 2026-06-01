"""K-Means Regime Strategy - clustering-based regime detection."""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from quant_portfolio.strategies.base import Strategy


class KMeansRegime(Strategy):
    """K-means regime clustering strategy.

    Clusters market conditions into regimes and trades based on
    which regime the market is currently in.

    Parameters
    ----------
    n_clusters : int
        Number of regimes (default 3).
    lookback : int
        Feature calculation window (default 60).
    """

    name = "kmeans_regime"

    def __init__(self, n_clusters: int = 3, lookback: int = 60, **kwargs):
        super().__init__(n_clusters=n_clusters, lookback=lookback, **kwargs)
        self.n_clusters = n_clusters
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on regime clustering."""
        close = data["close"]
        returns = close.pct_change()

        features = pd.DataFrame(index=data.index)
        features["ret_5"] = close.pct_change(5)
        features["ret_20"] = close.pct_change(20)
        features["vol"] = returns.rolling(20).std()
        features["skew"] = returns.rolling(20).skew()
        features = features.dropna()

        if len(features) < self.lookback:
            return pd.DataFrame({"signal": 0}, index=data.index)

        scaler = StandardScaler()
        X = scaler.fit_transform(features.values)

        km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        # Determine which cluster is bullish/bearish by average returns
        cluster_returns = {}
        for c in range(self.n_clusters):
            mask = labels == c
            cluster_returns[c] = features["ret_5"].values[mask].mean()

        best = max(cluster_returns, key=cluster_returns.get)
        worst = min(cluster_returns, key=cluster_returns.get)

        signal = pd.Series(0, index=data.index)
        for idx, label in zip(features.index, labels):
            if label == best:
                signal[idx] = 1
            elif label == worst:
                signal[idx] = -1

        return pd.DataFrame({"signal": signal}, index=data.index)
