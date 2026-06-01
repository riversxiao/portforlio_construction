"""Variational Inference algorithm."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_portfolio.algorithms.base import Algorithm


class VariationalInferenceAlgorithm(Algorithm):
    """Variational Inference for Portfolio Uncertainty.

    Approximates the posterior distribution of optimal weights using
    variational inference. Models weight uncertainty with a diagonal
    Gaussian variational distribution:

        q(w) = N(mu_w, diag(sigma_w^2))

    Optimizes the ELBO to find mean weights that account for parameter
    uncertainty.
    """

    name = "variational_inference"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu_ret = returns.mean().values
        cov_ret = returns.cov().values
        n_samples = self.params.get("n_samples", 50)
        rng = np.random.default_rng(42)

        # Variational parameters: mean and log-variance of weights
        mu_w = np.ones(n_assets) / n_assets
        log_sigma = np.full(n_assets, -3.0)

        lr = 0.01
        for _ in range(200):
            sigma = np.exp(log_sigma)
            # Sample from variational distribution
            eps = rng.normal(0, 1, (n_samples, n_assets))
            w_samples = mu_w + sigma * eps

            # Project each sample onto simplex (softmax approximation)
            for s in range(n_samples):
                w_samples[s] = np.maximum(w_samples[s], 0)
                total = w_samples[s].sum()
                if total > 1e-10:
                    w_samples[s] /= total

            # Expected negative Sharpe (loss)
            losses = []
            for s in range(n_samples):
                w = w_samples[s]
                port_ret = mu_ret @ w
                port_vol = np.sqrt(max(w @ cov_ret @ w, 1e-10))
                losses.append(-port_ret / port_vol)

            # Gradient estimate (REINFORCE)
            loss_mean = np.mean(losses)
            # Update mu_w toward better weights
            grad_mu = np.zeros(n_assets)
            for s in range(n_samples):
                grad_mu += (losses[s] - loss_mean) * eps[s]
            grad_mu /= n_samples

            mu_w -= lr * grad_mu
            mu_w = np.maximum(mu_w, 0)
            total = mu_w.sum()
            if total > 1e-10:
                mu_w /= total

        return mu_w
