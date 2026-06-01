"""Particle Swarm Optimization algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class ParticleSwarmAlgorithm(Algorithm):
    """Particle Swarm Optimization (PSO) for Portfolios.

    Swarm intelligence approach where particles move through weight
    space guided by personal best and global best positions:

        v_i = w*v_i + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
        x_i = x_i + v_i

    Minimizes portfolio variance while seeking positive returns.
    """

    name = "particle_swarm"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        n_particles = self.params.get("n_particles", 30)
        n_iter = self.params.get("n_iter", 100)
        rng = np.random.default_rng(42)

        def objective(w):
            port_vol = np.sqrt(w @ cov @ w)
            port_ret = mu @ w
            return port_vol - 0.5 * port_ret

        # Initialize
        positions = rng.dirichlet(np.ones(n_assets), n_particles)
        velocities = rng.normal(0, 0.01, (n_particles, n_assets))
        pbest = positions.copy()
        pbest_val = np.array([objective(p) for p in positions])
        gbest_idx = np.argmin(pbest_val)
        gbest = pbest[gbest_idx].copy()

        inertia = 0.7
        c1, c2 = 1.5, 1.5

        for _ in range(n_iter):
            r1 = rng.random((n_particles, n_assets))
            r2 = rng.random((n_particles, n_assets))

            velocities = (inertia * velocities +
                          c1 * r1 * (pbest - positions) +
                          c2 * r2 * (gbest - positions))
            positions = positions + velocities

            # Project onto simplex
            for i in range(n_particles):
                positions[i] = np.maximum(positions[i], 0)
                s = positions[i].sum()
                if s > 1e-10:
                    positions[i] /= s
                else:
                    positions[i] = np.ones(n_assets) / n_assets

                val = objective(positions[i])
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest[i] = positions[i].copy()

            gbest_idx = np.argmin(pbest_val)
            gbest = pbest[gbest_idx].copy()

        return gbest
