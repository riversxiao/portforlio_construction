"""Mean reversion and pairs trading example.

Demonstrates applying mean reversion strategies including pairs trading
on synthetic cointegrated data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_portfolio.backtesting.engine import Backtester
from quant_portfolio.strategies.registry import get_strategy


def generate_mean_reverting_data(n_periods: int = 504) -> pd.DataFrame:
    """Generate synthetic mean-reverting price data."""
    np.random.seed(123)
    dates = pd.date_range("2019-01-01", periods=n_periods, freq="B")

    # Ornstein-Uhlenbeck process
    theta = 0.1  # mean reversion speed
    mu = np.log(100)  # long-term mean
    sigma = 0.02  # volatility

    log_price = np.zeros(n_periods)
    log_price[0] = mu
    for t in range(1, n_periods):
        log_price[t] = (
            log_price[t - 1]
            + theta * (mu - log_price[t - 1])
            + sigma * np.random.normal()
        )

    close = np.exp(log_price)
    high = close * (1 + np.abs(np.random.normal(0, 0.008, n_periods)))
    low = close * (1 - np.abs(np.random.normal(0, 0.008, n_periods)))
    open_ = close * (1 + np.random.normal(0, 0.004, n_periods))
    volume = np.random.randint(500_000, 5_000_000, n_periods).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def main() -> None:
    """Run mean reversion strategy backtests."""
    data = generate_mean_reverting_data()
    print(f"Data: {len(data)} bars, mean-reverting process")
    print(f"Price range: {data['close'].min():.2f} to {data['close'].max():.2f}")
    print(f"Mean price: {data['close'].mean():.2f}\n")

    # Mean reversion strategies to test
    mr_strategies = [
        "bollinger_bands",
        "rsi_reversion",
        "zscore_reversion",
        "ornstein_uhlenbeck",
        "half_life_mr",
    ]

    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Sortino':>8}")
    print("-" * 65)

    for name in mr_strategies:
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
            f"{m['sortino_ratio']:>8.3f}"
        )

    print("\nMean reversion backtest complete.")


if __name__ == "__main__":
    main()
