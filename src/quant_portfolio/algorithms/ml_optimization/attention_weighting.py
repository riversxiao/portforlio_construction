"""Attention Mechanism Weighting algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class AttentionWeightingAlgorithm(Algorithm):
    """Attention Mechanism for Asset Weighting.

    Applies a self-attention mechanism to determine asset importance.
    Computes attention scores using scaled dot-product attention
    on asset feature vectors (recent returns):

        Attention(Q,K,V) = softmax(QK^T / sqrt(d)) * V

    The attention output determines portfolio weights.
    """

    name = "attention_weighting"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        lookback = min(self.params.get("lookback", 20), len(ret_matrix))

        # Feature matrix: recent returns per asset (n_assets x lookback)
        features = ret_matrix[-lookback:].T  # (n_assets, lookback)
        d_k = lookback

        # Self-attention: Q = K = V = features
        # Attention scores
        scores = features @ features.T / np.sqrt(d_k)

        # Softmax per row
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        attention = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        # Weighted features
        context = attention @ features

        # Aggregate: asset importance = norm of context vector
        importance = np.linalg.norm(context, axis=1)
        total = importance.sum()
        if total > 1e-10:
            weights = importance / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
