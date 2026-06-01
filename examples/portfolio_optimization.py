"""Portfolio optimization comparison example.

Demonstrates comparing multiple optimization algorithms on the same
set of assets to understand their different allocation approaches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.registry import get_algorithm


def generate_multi_asset_returns(
    n_assets: int = 10, n_periods: int = 504
) -> pd.DataFrame:
    """Generate synthetic multi-asset return data with realistic correlations."""
    np.random.seed(42)
    dates = pd.date_range("2019-01-01", periods=n_periods, freq="B")
    asset_names = [f"asset_{i:02d}" for i in range(n_assets)]

    # Create a correlation structure
    # Factor model: returns = beta * factor + idiosyncratic
    n_factors = 3
    factor_returns = np.random.normal(0.0003, 0.01, (n_periods, n_factors))
    betas = np.random.uniform(0.3, 1.5, (n_assets, n_factors))
    idiosyncratic = np.random.normal(0, 0.01, (n_periods, n_assets))

    # Add different expected returns for each asset
    expected_returns = np.linspace(0.0002, 0.001, n_assets)

    returns = factor_returns @ betas.T + idiosyncratic + expected_returns

    return pd.DataFrame(returns, index=dates, columns=asset_names)


def main() -> None:
    """Compare different optimization algorithms."""
    returns = generate_multi_asset_returns()
    print(f"Asset universe: {returns.shape[1]} assets, {returns.shape[0]} periods")
    print(f"Annualized returns range: "
          f"{returns.mean().min() * 252:.1%} to {returns.mean().max() * 252:.1%}\n")

    # Algorithms to compare
    algorithms = [
        "equal_weight",
        "min_variance",
        "max_sharpe",
        "risk_parity",
        "inverse_variance",
        "max_diversification",
        "hierarchical_risk_parity",
    ]

    print(f"{'Algorithm':<30} {'Max Wt':>8} {'Min Wt':>8} {'HHI':>8} {'ExpRet':>8}")
    print("-" * 70)

    for algo_name in algorithms:
        AlgoClass = get_algorithm(algo_name)
        algo = AlgoClass()
        weights = algo.optimize(returns)

        # Compute metrics
        max_wt = weights.max()
        min_wt = weights[weights > 0.001].min() if (weights > 0.001).any() else 0.0
        hhi = np.sum(weights**2)  # Herfindahl-Hirschman Index (concentration)
        exp_return = float(np.dot(weights, returns.mean()) * 252)

        print(
            f"{algo_name:<30} "
            f"{max_wt:>7.2%} "
            f"{min_wt:>7.2%} "
            f"{hhi:>8.4f} "
            f"{exp_return:>7.2%}"
        )

    print("\n--- Detailed weights for selected algorithms ---")
    for algo_name in ["equal_weight", "risk_parity", "max_sharpe"]:
        AlgoClass = get_algorithm(algo_name)
        algo = AlgoClass()
        weights = algo.optimize(returns)
        print(f"\n{algo_name}:")
        for i, w in enumerate(weights):
            if w > 0.001:
                print(f"  asset_{i:02d}: {w:.4f}")

    print("\nOptimization comparison complete.")


if __name__ == "__main__":
    main()
