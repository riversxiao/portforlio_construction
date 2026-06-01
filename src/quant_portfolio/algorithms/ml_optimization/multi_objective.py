"""Multi-Objective Optimization algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class MultiObjectiveAlgorithm(Algorithm):
    """Multi-Objective Optimization (Pareto Front).

    Optimizes portfolios considering multiple objectives simultaneously:
    1. Maximize return
    2. Minimize variance
    3. Minimize CVaR

    Finds portfolios on the Pareto front using weighted sum
    scalarization with multiple weight vectors, then selects
    the knee point (maximum distance from endpoints).
    """

    name = "multi_objective"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        mu = returns.mean().values
        cov = returns.cov().values
        ret_matrix = returns.values
        rng = np.random.default_rng(42)

        n_points = self.params.get("n_points", 50)

        def evaluate(w):
            ret = mu @ w
            var = w @ cov @ w
            sorted_r = np.sort(ret_matrix @ w)
            n_tail = max(1, int(len(sorted_r) * 0.05))
            cvar = -sorted_r[:n_tail].mean()
            return ret, var, cvar

        # Generate Pareto front candidates
        pareto_front = []
        for _ in range(n_points):
            w = rng.dirichlet(np.ones(n_assets))
            obj = evaluate(w)
            pareto_front.append((w, obj))

        # Find non-dominated solutions
        non_dominated = []
        for i, (w_i, obj_i) in enumerate(pareto_front):
            dominated = False
            for j, (w_j, obj_j) in enumerate(pareto_front):
                if i == j:
                    continue
                # j dominates i if better in all: higher ret, lower var, lower cvar
                if (obj_j[0] >= obj_i[0] and obj_j[1] <= obj_i[1] and
                        obj_j[2] <= obj_i[2] and
                        (obj_j[0] > obj_i[0] or obj_j[1] < obj_i[1] or obj_j[2] < obj_i[2])):
                    dominated = True
                    break
            if not dominated:
                non_dominated.append((w_i, obj_i))

        if not non_dominated:
            return np.ones(n_assets) / n_assets

        # Select knee point: best Sharpe-like ratio
        best_score = -np.inf
        best_w = non_dominated[0][0]
        for w, (ret, var, cvar) in non_dominated:
            vol = np.sqrt(max(var, 1e-10))
            score = ret / vol
            if score > best_score:
                best_score = score
                best_w = w

        return best_w
