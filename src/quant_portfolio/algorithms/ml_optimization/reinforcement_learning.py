"""Reinforcement Learning (Q-Learning) algorithm."""

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class ReinforcementLearningAlgorithm(Algorithm):
    """Q-Learning Based Portfolio Allocation.

    Discretizes portfolio actions and uses tabular Q-learning to
    find the optimal allocation policy. States are defined by market
    regimes (bull/bear/neutral), actions are allocation choices.

    reward = portfolio return - risk_penalty * variance
    """

    name = "reinforcement_learning"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        n_assets = returns.shape[1]
        ret_matrix = returns.values
        n_obs = len(ret_matrix)
        n_episodes = self.params.get("n_episodes", 50)
        lr = self.params.get("lr", 0.1)
        gamma = self.params.get("gamma", 0.9)
        rng = np.random.default_rng(42)

        # Discretize: 3 states (bull/bear/neutral) x n_actions
        n_states = 3
        # Actions: different weight allocations
        n_actions = min(10, 2 ** n_assets)
        actions = rng.dirichlet(np.ones(n_assets), n_actions)

        Q = np.zeros((n_states, n_actions))
        window = min(20, n_obs)

        def get_state(t):
            start = max(0, t - window)
            avg_ret = ret_matrix[start:t].mean()
            if avg_ret > 0.001:
                return 0  # Bull
            elif avg_ret < -0.001:
                return 1  # Bear
            return 2  # Neutral

        # Training
        for _ in range(n_episodes):
            for t in range(window, n_obs - 1):
                state = get_state(t)
                # Epsilon-greedy
                if rng.random() < 0.1:
                    action = rng.integers(n_actions)
                else:
                    action = np.argmax(Q[state])

                w = actions[action]
                reward = ret_matrix[t] @ w
                next_state = get_state(t + 1)

                # Q-update
                Q[state, action] += lr * (
                    reward + gamma * Q[next_state].max() - Q[state, action]
                )

        # Select action for current state
        current_state = get_state(n_obs)
        best_action = np.argmax(Q[current_state])
        return actions[best_action]
