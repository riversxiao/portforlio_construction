"""Correlation Regime Detection algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class CorrelationRegimeAlgorithm(Algorithm):
    """Correlation Regime Detection and Allocation.

    Detects the current correlation regime (high/low correlation)
    and adjusts portfolio weights accordingly. In high-correlation
    regimes, diversification benefits diminish, so the portfolio
    shifts toward minimum variance. In low-correlation regimes,
    it increases diversification.

    Regime detection uses rolling average pairwise correlation.
    """

    name = "correlation_regime"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        cov = returns.cov().values
        corr = returns.corr().values

        # Average pairwise correlation
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        avg_corr = corr[mask].mean() if mask.sum() > 0 else 0.5
        threshold = self.params.get("corr_threshold", 0.5)

        if avg_corr > threshold:
            # High correlation regime: minimum variance
            vols = np.sqrt(np.diag(cov))
            inv_vol = 1.0 / np.maximum(vols, 1e-10)
            weights = inv_vol / inv_vol.sum()
        else:
            # Low correlation regime: maximize diversification
            vols = np.sqrt(np.diag(cov))
            # Weight by contribution to diversification
            inv_corr_sum = np.zeros(n_assets)
            for i in range(n_assets):
                inv_corr_sum[i] = 1.0 / (np.abs(corr[i]).mean() + 1e-10)
            weights = inv_corr_sum / inv_corr_sum.sum()

        return weights
