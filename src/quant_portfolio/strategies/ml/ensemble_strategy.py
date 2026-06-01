"""Ensemble Strategy - combination of multiple ML models."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from quant_portfolio.strategies.base import Strategy


class EnsembleStrategy(Strategy):
    """Ensemble of multiple ML models strategy.

    Combines predictions from Random Forest, Gradient Boosting, and
    Logistic Regression via majority voting.

    Parameters
    ----------
    lookback : int
        Training window (default 120).
    """

    name = "ensemble_strategy"

    def __init__(self, lookback: int = 120, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.lookback = lookback

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
        """Generate signals from ensemble majority voting."""
        features = self._build_features(data)
        close = data["close"]
        target = np.sign(close.pct_change().shift(-1))

        signal = pd.Series(0, index=data.index)
        step = max(20, self.lookback // 4)

        for i in range(self.lookback, len(data) - 1, step):
            end_pred = min(i + step, len(data))
            X_train = features.iloc[i - self.lookback:i].dropna()
            y_train = target.iloc[i - self.lookback:i].loc[X_train.index]
            mask = ~(y_train.isna() | (y_train == 0))
            X_train = X_train[mask]
            y_train = y_train[mask]

            if len(X_train) < 20:
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train.values)

            models = [
                RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42),
                GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42),
                LogisticRegression(random_state=42, max_iter=200),
            ]
            for m in models:
                m.fit(X_scaled, y_train.values.astype(int))

            X_pred = features.iloc[i:end_pred].dropna()
            if len(X_pred) > 0:
                X_pred_scaled = scaler.transform(X_pred.values)
                votes = np.array([m.predict(X_pred_scaled) for m in models])
                ensemble_pred = np.sign(votes.sum(axis=0))
                signal.loc[X_pred.index] = ensemble_pred

        return pd.DataFrame({"signal": signal}, index=data.index)
