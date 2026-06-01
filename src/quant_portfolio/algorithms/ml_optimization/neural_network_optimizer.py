"""Neural Network Optimizer algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class NeuralNetworkOptimizerAlgorithm(Algorithm):
    """Simple Neural Network for Weight Optimization.

    A single hidden layer neural network that maps recent return
    features to portfolio weights. Uses gradient descent to train
    the network to minimize negative Sharpe ratio.

    Architecture: input(n_features) -> hidden(16) -> output(n_assets) -> softmax
    """

    name = "neural_network_optimizer"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_obs = len(ret_matrix)
        hidden_size = self.params.get("hidden_size", 16)
        n_epochs = self.params.get("n_epochs", 100)
        lr = self.params.get("learning_rate", 0.01)
        rng = np.random.default_rng(42)

        # Features: rolling mean and vol for each asset
        window = min(20, n_obs)
        features = np.concatenate([
            ret_matrix[-window:].mean(axis=0),
            ret_matrix[-window:].std(axis=0),
        ])
        n_features = len(features)

        # Initialize network
        W1 = rng.normal(0, 0.1, (n_features, hidden_size))
        b1 = np.zeros(hidden_size)
        W2 = rng.normal(0, 0.1, (hidden_size, n_assets))
        b2 = np.zeros(n_assets)

        mu = ret_matrix.mean(axis=0)
        cov = np.cov(ret_matrix, rowvar=False)

        for _ in range(n_epochs):
            # Forward pass
            h = np.maximum(features @ W1 + b1, 0)  # ReLU
            logits = h @ W2 + b2
            # Softmax for weights
            exp_logits = np.exp(logits - logits.max())
            weights = exp_logits / exp_logits.sum()

            # Loss: negative Sharpe
            port_ret = mu @ weights
            port_vol = np.sqrt(max(weights @ cov @ weights, 1e-10))
            loss = -port_ret / port_vol

            # Backprop (approximate with finite differences on logits)
            grad_logits = np.zeros(n_assets)
            eps = 1e-4
            for j in range(n_assets):
                logits_p = logits.copy()
                logits_p[j] += eps
                exp_p = np.exp(logits_p - logits_p.max())
                w_p = exp_p / exp_p.sum()
                ret_p = mu @ w_p
                vol_p = np.sqrt(max(w_p @ cov @ w_p, 1e-10))
                loss_p = -ret_p / vol_p
                grad_logits[j] = (loss_p - loss) / eps

            # Update W2 and b2
            W2 -= lr * np.outer(h, grad_logits)
            b2 -= lr * grad_logits

        # Final forward pass
        h = np.maximum(features @ W1 + b1, 0)
        logits = h @ W2 + b2
        exp_logits = np.exp(logits - logits.max())
        weights = exp_logits / exp_logits.sum()
        return weights
