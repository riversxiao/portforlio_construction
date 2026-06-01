"""Multi-strategy portfolio construction example.

Demonstrates combining multiple trading strategies into a single portfolio
with optimized weights using the PortfolioConstructor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.registry import get_algorithm
from quant_portfolio.portfolio.constructor import PortfolioConstructor
from quant_portfolio.strategies.registry import get_strategy


def generate_multi_asset_data(
    n_assets: int = 5, n_periods: int = 504
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Generate synthetic OHLCV data for multiple assets."""
    np.random.seed(55)
    dates = pd.date_range("2019-01-01", periods=n_periods, freq="B")
    asset_names = [f"stock_{i}" for i in range(n_assets)]

    datasets = []
    returns_list = []

    for i in range(n_assets):
        drift = np.random.uniform(0.0002, 0.0008)
        vol = np.random.uniform(0.015, 0.03)
        close = 100 * np.exp(
            np.cumsum(np.random.normal(drift, vol, n_periods))
        )
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
        open_ = close * (1 + np.random.normal(0, 0.005, n_periods))
        volume = np.random.randint(500_000, 8_000_000, n_periods).astype(float)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
        datasets.append(df)
        returns_list.append(
            pd.Series(close, index=dates).pct_change().rename(asset_names[i])
        )

    returns_df = pd.concat(returns_list, axis=1).dropna()
    return datasets, returns_df


def main() -> None:
    """Build a multi-strategy portfolio."""
    n_assets = 5
    asset_names = [f"stock_{i}" for i in range(n_assets)]
    datasets, returns_df = generate_multi_asset_data(n_assets=n_assets)
    dates = returns_df.index

    print(f"Assets: {n_assets}, Periods: {len(returns_df)}")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}\n")

    # Apply different strategies to different assets
    strategy_assignments = [
        "simple_momentum",
        "bollinger_bands",
        "macd_momentum",
        "rsi_reversion",
        "moving_average_crossover",
    ]

    signals_list = []
    print("Strategy assignments:")
    for i, (data, strat_name) in enumerate(zip(datasets, strategy_assignments)):
        StrategyClass = get_strategy(strat_name)
        strategy = StrategyClass()
        sigs = strategy.generate_signals(data)
        signals_list.append(sigs["signal"].rename(asset_names[i]))
        buy_pct = (sigs["signal"] == 1).mean()
        print(f"  {asset_names[i]} -> {strat_name} (buy signals: {buy_pct:.1%})")

    signals_df = pd.concat(signals_list, axis=1).reindex(returns_df.index).fillna(0)

    # Construct portfolio with different algorithms
    print(f"\n{'Algorithm':<30} {'Ann Return':>10} {'Ann Vol':>8} {'Sharpe':>8}")
    print("-" * 60)

    for algo_name in ["equal_weight", "risk_parity", "min_variance", "inverse_variance"]:
        AlgoClass = get_algorithm(algo_name)
        algo = AlgoClass()

        constructor = PortfolioConstructor(
            algorithm=algo,
            rebalance_frequency="monthly",
        )
        weights = constructor.construct(signals_df, returns_df)

        # Calculate portfolio returns
        port_returns = (weights * returns_df).sum(axis=1)
        ann_return = port_returns.mean() * 252
        ann_vol = port_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        print(f"{algo_name:<30} {ann_return:>9.2%} {ann_vol:>7.2%} {sharpe:>8.3f}")

    # Show final weights for the best approach
    print("\n--- Final weights (risk_parity, monthly rebalance) ---")
    AlgoClass = get_algorithm("risk_parity")
    algo = AlgoClass()
    constructor = PortfolioConstructor(algorithm=algo, rebalance_frequency="monthly")
    weights = constructor.construct(signals_df, returns_df)
    last_weights = weights.iloc[-1]
    for asset, w in last_weights.items():
        if w > 0.001:
            print(f"  {asset}: {w:.4f}")
    print(f"  Total: {last_weights.sum():.4f}")

    print("\nMulti-strategy portfolio construction complete.")


if __name__ == "__main__":
    main()
