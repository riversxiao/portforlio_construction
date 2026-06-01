"""Logistic Regression Strategy - probability-based signals."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler

from quant_portfolio.strategies.base import Strategy


class LogisticRegressionStrategy(Strategy):
    """Logistic regression probability strategy.

    Uses logistic regression to estimate the probability of positive
    next-day returns from technical features.

    Parameters
    ----------
    lookback : int
        Training window (default 120).
    threshold : float
        Probability threshold for signal (default 0.6).
    """

    name = "logistic_regression"

    def __init__(self, lookback: int = 120, threshold: float = 0.6, **kwargs):
        super().__init__(lookback=lookback, threshold=threshold, **kwargs)
        self.lookback = lookback
        self.threshold = threshold

    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        features = pd.DataFrame(index=data.index)
        features["ret_1"] = close.pct_change(1)
        features["ret_5"] = close.pct_change(5)
        features["ret_10"] = close.pct_change(10)
        features["vol"] = close.pct_change().rolling(10).std()
        features["ma_ratio"] = close / (close.rolling(20).mean() + 1e-10)
        return features

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using logistic regression probabilities."""
        features = self._build_features(data)
        close = data["close"]
        target = (close.pct_change().shift(-1) > 0).astype(int)

        signal = pd.Series(0, index=data.index)
        step = max(20, self.lookback // 4)

        for i in range(self.lookback, len(data) - 1, step):
            end_pred = min(i + step, len(data))
            X_train = features.iloc[i - self.lookback:i].dropna()
            y_train = target.iloc[i - self.lookback:i].loc[X_train.index].dropna()
            common = X_train.index.intersection(y_train.index)
            X_train = X_train.loc[common]
            y_train = y_train.loc[common]

            if len(X_train) < 20:
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train.values)
            clf = LR(random_state=42, max_iter=200)
            clf.fit(X_scaled, y_train.values)

            X_pred = features.iloc[i:end_pred].dropna()
            if len(X_pred) > 0:
                probs = clf.predict_proba(scaler.transform(X_pred.values))[:, 1]
                preds = np.where(probs > self.threshold, 1,
                                 np.where(probs < 1 - self.threshold, -1, 0))
                signal.loc[X_pred.index] = preds

        return pd.DataFrame({"signal": signal}, index=data.index)
