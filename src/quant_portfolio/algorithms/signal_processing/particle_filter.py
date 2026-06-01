"""Particle Filter State Estimation algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class ParticleFilterAlgorithm(Algorithm):
    """Particle Filter for State Estimation and Allocation.

    Uses Sequential Monte Carlo (particle filter) to estimate the
    hidden state (true trend) of each asset's return process.
    Particles represent possible states; their weighted average
    gives the state estimate.

    State model: x_t = x_{t-1} + process_noise
    Observation: y_t = x_t + observation_noise
    """

    name = "particle_filter"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_particles = self.params.get("n_particles", 100)
        process_std = self.params.get("process_std", 0.005)
        obs_std = self.params.get("obs_std", 0.02)
        rng = np.random.default_rng(42)

        trend_estimates = np.zeros(n_assets)
        for i in range(n_assets):
            observations = ret_matrix[:, i]
            # Initialize particles
            particles = rng.normal(0, obs_std, n_particles)
            weights_pf = np.ones(n_particles) / n_particles

            for obs in observations:
                # Propagate
                particles = particles + rng.normal(0, process_std, n_particles)
                # Weight by likelihood
                likelihood = np.exp(-0.5 * ((obs - particles) / obs_std) ** 2)
                weights_pf = weights_pf * likelihood
                w_sum = weights_pf.sum()
                if w_sum > 1e-20:
                    weights_pf = weights_pf / w_sum
                else:
                    weights_pf = np.ones(n_particles) / n_particles

                # Resample if effective sample size is low
                n_eff = 1.0 / (weights_pf ** 2).sum()
                if n_eff < n_particles / 2:
                    indices = rng.choice(n_particles, size=n_particles, p=weights_pf)
                    particles = particles[indices]
                    weights_pf = np.ones(n_particles) / n_particles

            # State estimate
            trend_estimates[i] = (weights_pf * particles).sum()

        scores = np.maximum(trend_estimates, 0)
        total = scores.sum()
        if total > 1e-10:
            weights = scores / total
        else:
            weights = np.ones(n_assets) / n_assets
        return weights
