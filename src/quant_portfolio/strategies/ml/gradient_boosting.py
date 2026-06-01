"""Gradient Boosting Strategy - GBM classification signals."""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from quant_portfolio.strategies.base import Strategy


class GradientBoosting(Strategy):
    """Gradient Boosting classification strategy.

    Uses gradient boosting on technical features to predict direction.

    Parameters
    ----------
    lookback : int
        Training window (default 120).
    n_estimators : int
        Number of boosting stages (default 50).
    """

    name = "gradient_boosting"

    def __init__(self, lookback: int = 120, n_estimators: int = 50, **kwargs):
        super().__init__(lookback=lookback, n_estimators=n_estimators, **kwargs)
        self.lookback = lookback
        self.n_estimators = n_estimators

    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        features = pd.DataFrame(index=data.index)
        features["ret_1"] = close.pct_change(1)
        features["ret_5"] = close.pct_change(5)
        features["ret_20"] = close.pct_change(20)
        features["ma_ratio"] = close / (close.rolling(20).mean() + 1e-10)
        features["vol"] = close.pct_change().rolling(10).std()
        return features

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using Gradient Boosting predictions."""
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

            clf = GradientBoostingClassifier(
                n_estimators=self.n_estimators, max_depth=3, random_state=42
            )
            clf.fit(X_train.values, y_train.values.astype(int))

            X_pred = features.iloc[i:end_pred].dropna()
            if len(X_pred) > 0:
                preds = clf.predict(X_pred.values)
                signal.loc[X_pred.index] = preds

        return pd.DataFrame({"signal": signal}, index=data.index)
