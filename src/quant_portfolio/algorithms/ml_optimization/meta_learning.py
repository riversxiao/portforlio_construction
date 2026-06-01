"""Meta-Learning algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class MetaLearningAlgorithm(Algorithm):
    """Meta-Learning Across Market Regimes.

    Implements a simplified MAML-like approach: learns allocation
    strategies across different market regimes (sub-periods) and
    finds weights that perform well across all regimes.

    Splits history into episodes, optimizes per-episode, then
    averages (meta-gradient direction).
    """

    name = "meta_learning"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_obs = len(ret_matrix)
        n_episodes = self.params.get("n_episodes", 5)
        episode_len = n_obs // max(n_episodes, 1)

        if episode_len < 5:
            return np.ones(n_assets) / n_assets

        # Meta-weights: average of per-episode optimal weights
        all_weights = []
        for ep in range(n_episodes):
            start = ep * episode_len
            end = min(start + episode_len, n_obs)
            episode_ret = ret_matrix[start:end]

            if len(episode_ret) < 3:
                continue

            # Per-episode inverse variance
            var_ep = episode_ret.var(axis=0)
            var_ep = np.maximum(var_ep, 1e-10)
            inv_var = 1.0 / var_ep
            w_ep = inv_var / inv_var.sum()
            all_weights.append(w_ep)

        if all_weights:
            return np.mean(all_weights, axis=0)
        return np.ones(n_assets) / n_assets
