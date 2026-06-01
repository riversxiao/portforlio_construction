"""SVM Strategy - Support Vector Machine classification."""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from quant_portfolio.strategies.base import Strategy


class SVMStrategy(Strategy):
    """SVM classification strategy.

    Uses Support Vector Machine with RBF kernel on technical features.

    Parameters
    ----------
    lookback : int
        Training window (default 120).
    C : float
        Regularization parameter (default 1.0).
    """

    name = "svm_strategy"

    def __init__(self, lookback: int = 120, C: float = 1.0, **kwargs):
        super().__init__(lookback=lookback, C=C, **kwargs)
        self.lookback = lookback
        self.C = C

    def _build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        features = pd.DataFrame(index=data.index)
        features["ret_1"] = close.pct_change(1)
        features["ret_5"] = close.pct_change(5)
        features["ret_10"] = close.pct_change(10)
        features["vol"] = close.pct_change().rolling(10).std()
        features["ma_dist"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-10)
        return features

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using SVM predictions."""
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
            clf = SVC(C=self.C, kernel="rbf", random_state=42)
            clf.fit(X_scaled, y_train.values.astype(int))

            X_pred = features.iloc[i:end_pred].dropna()
            if len(X_pred) > 0:
                preds = clf.predict(scaler.transform(X_pred.values))
                signal.loc[X_pred.index] = preds

        return pd.DataFrame({"signal": signal}, index=data.index)
