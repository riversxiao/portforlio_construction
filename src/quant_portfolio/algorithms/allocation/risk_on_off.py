"""Risk-On/Risk-Off Regime Allocation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class RiskOnOffAlgorithm(Algorithm):
    """Risk-On / Risk-Off Regime Allocation.

    Classifies the current market as risk-on or risk-off based on
    realized volatility relative to its historical average:

    - Risk-on (low vol): overweight high-return assets
    - Risk-off (high vol): overweight low-volatility assets

    Regime detection uses rolling volatility z-score.
    """

    name = "risk_on_off"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        vol_lookback = self.params.get("vol_lookback", 21)
        threshold = self.params.get("threshold", 0.0)

        # Portfolio-level volatility
        n_obs = len(ret_matrix)
        window = min(vol_lookback, n_obs)
        port_ret = ret_matrix.mean(axis=1)

        recent_vol = port_ret[-window:].std()
        long_vol = port_ret.std()

        vol_zscore = (recent_vol - long_vol) / max(long_vol, 1e-10)

        vols = returns.std().values
        mus = returns.mean().values

        if vol_zscore < threshold:
            # Risk-on: weight by Sharpe ratio
            sharpe = mus / np.maximum(vols, 1e-10)
            scores = np.maximum(sharpe, 0)
        else:
            # Risk-off: inverse volatility
            scores = 1.0 / np.maximum(vols, 1e-10)

        total = scores.sum()
        if total > 1e-10:
            weights = scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
