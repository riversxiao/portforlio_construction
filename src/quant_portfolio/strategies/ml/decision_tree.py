"""Decision Tree Strategy - tree-based rules."""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from quant_portfolio.strategies.base import Strategy


class DecisionTreeStrategy(Strategy):
    """Decision tree classification strategy.

    Uses a shallow decision tree for interpretable trading rules.

    Parameters
    ----------
    lookback : int
        Training window (default 120).
    max_depth : int
        Maximum tree depth (default 4).
    """

    name = "decision_tree"

    def __init__(self, lookback: int = 120, max_depth: int = 4, **kwargs):
        super().__init__(lookback=lookback, max_depth=max_depth, **kwargs)
        self.lookback = lookback
        self.max_depth = max_depth

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
        """Generate signals using Decision Tree predictions."""
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

            clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
            clf.fit(X_train.values, y_train.values.astype(int))

            X_pred = features.iloc[i:end_pred].dropna()
            if len(X_pred) > 0:
                preds = clf.predict(X_pred.values)
                signal.loc[X_pred.index] = preds

        return pd.DataFrame({"signal": signal}, index=data.index)
