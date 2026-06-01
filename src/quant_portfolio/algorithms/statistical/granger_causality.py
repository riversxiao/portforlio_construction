"""Granger Causality algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class GrangerCausalityAlgorithm(Algorithm):
    """Granger Causality Testing for Lead-Lag Allocation.

    Tests pairwise Granger causality between assets to identify
    leading indicators. Assets that Granger-cause more other assets
    are likely to be information leaders and receive higher weights.

    Uses simple VAR(1) F-test approximation.
    """

    name = "granger_causality"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_obs = len(ret_matrix)
        lag = self.params.get("lag", 1)

        if n_obs < lag + 5:
            return np.ones(n_assets) / n_assets

        # Count how many assets each one Granger-causes
        lead_scores = np.zeros(n_assets)

        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    continue
                # Test if i Granger-causes j
                Y = ret_matrix[lag:, j]
                X_restricted = ret_matrix[:-lag, j].reshape(-1, 1)
                X_full = np.column_stack([
                    ret_matrix[:-lag, j],
                    ret_matrix[:-lag, i]
                ])

                n = len(Y)
                # Restricted model RSS
                beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
                rss_r = np.sum((Y - X_restricted @ beta_r) ** 2)

                # Full model RSS
                beta_f = np.linalg.lstsq(X_full, Y, rcond=None)[0]
                rss_f = np.sum((Y - X_full @ beta_f) ** 2)

                # F-statistic
                p = 1  # one extra regressor
                if rss_f > 1e-10:
                    f_stat = ((rss_r - rss_f) / p) / (rss_f / (n - 2 * p))
                    # Significant if F > 4 (approx 5% level)
                    if f_stat > 4:
                        lead_scores[i] += 1

        # Weight by leadership score
        scores = lead_scores + 1  # Add 1 to avoid zero weights
        weights = scores / scores.sum()
        return weights
