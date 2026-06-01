"""Momentum strategies backtest example.

Demonstrates backtesting multiple momentum strategies on synthetic data
and comparing their performance metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_portfolio.backtesting.engine import Backtester
from quant_portfolio.strategies.registry import get_strategy


def generate_trending_data(n_periods: int = 504) -> pd.DataFrame:
    """Generate synthetic data with a trending component."""
    np.random.seed(42)
    dates = pd.date_range("2019-01-01", periods=n_periods, freq="B")

    # Create a trend + noise process
    trend = np.linspace(0, 0.5, n_periods)
    noise = np.cumsum(np.random.normal(0, 0.015, n_periods))
    log_prices = trend + noise
    close = 100 * np.exp(log_prices)

    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
    open_ = close * (1 + np.random.normal(0, 0.005, n_periods))
    volume = np.random.randint(1_000_000, 10_000_000, n_periods).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def main() -> None:
    """Backtest multiple momentum strategies and compare results."""
    data = generate_trending_data()
    print(f"Data: {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Price range: {data['close'].min():.2f} to {data['close'].max():.2f}\n")

    # Momentum strategies to test
    momentum_strategies = [
        "simple_momentum",
        "macd_momentum",
        "rsi_momentum",
        "dual_momentum",
        "time_series_momentum",
    ]

    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8}")
    print("-" * 65)

    for name in momentum_strategies:
        StrategyClass = get_strategy(name)
        strategy = StrategyClass()

        backtester = Backtester(
            strategy=strategy,
            initial_capital=1_000_000,
            commission=0.001,
            slippage=0.0005,
        )
        result = backtester.run(data)

        m = result.metrics
        print(
            f"{name:<25} "
            f"{m['total_return']:>9.2%} "
            f"{m['sharpe_ratio']:>8.3f} "
            f"{m['max_drawdown']:>8.2%} "
            f"{m['calmar_ratio']:>8.3f}"
        )

    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
