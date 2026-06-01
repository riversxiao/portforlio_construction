"""Tail Risk (Extreme Value Theory) algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class TailRiskAlgorithm(Algorithm):
    """Tail Risk Measurement and Allocation (EVT-based).

    Uses Extreme Value Theory concepts to measure tail risk.
    Estimates the tail index (shape parameter) for each asset
    using the Hill estimator, then allocates inversely to tail
    heaviness:

        w_i = (1/xi_i) / sum(1/xi_j)

    where xi_i is the tail index (higher = heavier tail = riskier).
    """

    name = "tail_risk"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        k_pct = self.params.get("tail_fraction", 0.1)

        tail_indices = np.zeros(n_assets)
        for i in range(n_assets):
            losses = -ret_matrix[:, i]
            sorted_losses = np.sort(losses)[::-1]
            k = max(2, int(len(sorted_losses) * k_pct))
            # Hill estimator for tail index
            top_k = sorted_losses[:k]
            threshold = sorted_losses[k] if k < len(sorted_losses) else sorted_losses[-1]
            if threshold > 0:
                log_ratios = np.log(top_k / threshold)
                xi = log_ratios.mean()
            else:
                xi = 1.0
            tail_indices[i] = max(xi, 0.01)

        # Inverse tail index weighting
        inv_tail = 1.0 / tail_indices
        weights = inv_tail / inv_tail.sum()
        return weights
