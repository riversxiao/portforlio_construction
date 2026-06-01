"""Liquidity Risk algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class LiquidityRiskAlgorithm(Algorithm):
    """Liquidity-Adjusted Risk Allocation.

    Estimates liquidity using the Amihud illiquidity measure
    (absolute return / volume proxy). Assets with lower liquidity
    receive lower weights to reduce liquidation risk:

        Illiquidity_i = mean(|r_i| / volume_i)

    Since volume data may not be available, uses return autocorrelation
    as a liquidity proxy (illiquid assets have higher autocorrelation).
    """

    name = "liquidity_risk"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values

        # Use absolute return magnitude and autocorrelation as liquidity proxy
        liquidity_scores = np.zeros(n_assets)
        for i in range(n_assets):
            series = ret_matrix[:, i]
            # Autocorrelation(1) as illiquidity proxy
            if len(series) > 1:
                autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                autocorr = abs(autocorr) if not np.isnan(autocorr) else 0.5
            else:
                autocorr = 0.5
            # Lower autocorrelation = more liquid = higher score
            liquidity_scores[i] = 1.0 - autocorr

        liquidity_scores = np.maximum(liquidity_scores, 0.01)
        weights = liquidity_scores / liquidity_scores.sum()
        return weights
