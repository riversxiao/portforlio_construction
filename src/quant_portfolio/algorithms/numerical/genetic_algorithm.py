"""Genetic Algorithm for Portfolio Optimization."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class GeneticAlgorithm(Algorithm):
    """Genetic Algorithm for Portfolio Optimization.

    Evolutionary optimization using selection, crossover, and mutation:
    1. Initialize population of random weight vectors
    2. Evaluate fitness (negative Sharpe ratio)
    3. Select parents (tournament selection)
    4. Crossover and mutate to create offspring
    5. Repeat for N generations
    """

    name = "genetic_algorithm"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        pop_size = self.params.get("pop_size", 50)
        n_gen = self.params.get("n_generations", 100)
        mut_rate = self.params.get("mutation_rate", 0.1)
        rng = np.random.default_rng(42)

        def fitness(w):
            port_ret = mu @ w
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 0.0
            return port_ret / port_vol

        # Initialize population on simplex
        pop = rng.dirichlet(np.ones(n_assets), pop_size)
        best_w = pop[0].copy()
        best_fit = fitness(best_w)

        for _ in range(n_gen):
            fits = np.array([fitness(ind) for ind in pop])
            # Update best
            max_idx = np.argmax(fits)
            if fits[max_idx] > best_fit:
                best_fit = fits[max_idx]
                best_w = pop[max_idx].copy()

            # Tournament selection and crossover
            new_pop = [best_w.copy()]  # Elitism
            for _ in range(pop_size - 1):
                # Tournament
                i1, i2 = rng.integers(0, pop_size, 2)
                p1 = pop[i1] if fits[i1] > fits[i2] else pop[i2]
                i3, i4 = rng.integers(0, pop_size, 2)
                p2 = pop[i3] if fits[i3] > fits[i4] else pop[i4]

                # Crossover
                alpha = rng.random()
                child = alpha * p1 + (1 - alpha) * p2

                # Mutation
                if rng.random() < mut_rate:
                    child += rng.normal(0, 0.02, n_assets)
                    child = np.maximum(child, 0)

                child = child / child.sum()
                new_pop.append(child)

            pop = np.array(new_pop)

        return best_w
