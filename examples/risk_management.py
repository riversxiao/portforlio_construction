"""Risk management demonstration example.

Shows how to use risk analysis utilities and risk-based algorithms
to manage portfolio risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.registry import get_algorithm
from quant_portfolio.portfolio.risk import (
    calculate_beta,
    calculate_cvar,
    calculate_tracking_error,
    calculate_var,
    calculate_volatility,
)


def generate_portfolio_data(n_periods: int = 504) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Generate synthetic portfolio, benchmark, and multi-asset return data."""
    np.random.seed(99)
    dates = pd.date_range("2019-01-01", periods=n_periods, freq="B")

    # Market factor
    market_returns = np.random.normal(0.0004, 0.012, n_periods)

    # Portfolio with beta to market
    beta = 1.2
    alpha = 0.0002
    portfolio_returns = alpha + beta * market_returns + np.random.normal(0, 0.008, n_periods)

    # Benchmark (market)
    benchmark_returns = market_returns

    # Multi-asset returns for optimization
    n_assets = 8
    asset_names = [f"asset_{i}" for i in range(n_assets)]
    betas = np.random.uniform(0.5, 1.5, n_assets)
    asset_returns = np.outer(market_returns, betas) + np.random.normal(
        0, 0.01, (n_periods, n_assets)
    )
    # Add small drift
    asset_returns += np.random.uniform(0.0001, 0.0005, n_assets)

    port_series = pd.Series(portfolio_returns, index=dates, name="portfolio")
    bench_series = pd.Series(benchmark_returns, index=dates, name="benchmark")
    multi_returns = pd.DataFrame(asset_returns, index=dates, columns=asset_names)

    return port_series, bench_series, multi_returns


def main() -> None:
    """Demonstrate risk management tools and algorithms."""
    port_returns, bench_returns, multi_returns = generate_portfolio_data()

    print("=" * 60)
    print("RISK ANALYSIS REPORT")
    print("=" * 60)

    # Basic risk metrics
    print("\n--- Portfolio Risk Metrics ---")
    var_95 = calculate_var(port_returns, confidence=0.95, method="historical")
    var_99 = calculate_var(port_returns, confidence=0.99, method="historical")
    cvar_95 = calculate_cvar(port_returns, confidence=0.95)
    ann_vol = calculate_volatility(port_returns, annualize=True)
    beta = calculate_beta(port_returns, bench_returns)
    te = calculate_tracking_error(port_returns, bench_returns, annualize=True)

    print(f"  Annualized Volatility: {ann_vol:.2%}")
    print(f"  95% VaR (daily):       {var_95:.4f}")
    print(f"  99% VaR (daily):       {var_99:.4f}")
    print(f"  95% CVaR (daily):      {cvar_95:.4f}")
    print(f"  Portfolio Beta:         {beta:.3f}")
    print(f"  Tracking Error (ann):   {te:.2%}")

    # Parametric vs Historical VaR comparison
    print("\n--- VaR Method Comparison ---")
    var_hist = calculate_var(port_returns, confidence=0.95, method="historical")
    var_param = calculate_var(port_returns, confidence=0.95, method="parametric")
    print(f"  Historical 95% VaR:  {var_hist:.4f}")
    print(f"  Parametric 95% VaR:  {var_param:.4f}")
    print(f"  Difference:          {abs(var_hist - var_param):.4f}")

    # Risk-based portfolio optimization
    print("\n--- Risk-Based Portfolio Optimization ---")
    risk_algorithms = [
        "var_historical",
        "cvar_calculation",
        "drawdown_control",
        "risk_budgeting",
        "concentration_risk",
    ]

    print(f"{'Algorithm':<25} {'Max Wt':>8} {'Min Wt':>8} {'Port Vol':>9}")
    print("-" * 55)

    for algo_name in risk_algorithms:
        AlgoClass = get_algorithm(algo_name)
        algo = AlgoClass()
        weights = algo.optimize(multi_returns)

        # Portfolio volatility with these weights
        port_vol = float(
            np.sqrt(
                weights @ multi_returns.cov().values @ weights
            )
        ) * np.sqrt(252)
        max_wt = weights.max()
        min_wt = weights[weights > 0.001].min() if (weights > 0.001).any() else 0.0

        print(f"  {algo_name:<23} {max_wt:>7.2%} {min_wt:>7.2%} {port_vol:>8.2%}")

    # Drawdown analysis
    print("\n--- Drawdown Analysis ---")
    cumulative = (1 + port_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    avg_dd = drawdown[drawdown < 0].mean()

    print(f"  Maximum Drawdown: {max_dd:.2%} (on {max_dd_date.date()})")
    print(f"  Average Drawdown: {avg_dd:.2%}")
    print(f"  Time in Drawdown: {(drawdown < 0).mean():.1%} of periods")

    print("\n" + "=" * 60)
    print("Risk management analysis complete.")


if __name__ == "__main__":
    main()
