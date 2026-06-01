"""Quick start example: end-to-end portfolio construction workflow.

This example demonstrates:
1. Generating sample data (in place of fetching from Baostock)
2. Applying a simple moving average crossover strategy
3. Running a backtest
4. Constructing a portfolio with equal-weight optimization
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm
from quant_portfolio.backtesting.engine import Backtester
from quant_portfolio.portfolio.constructor import PortfolioConstructor
from quant_portfolio.strategies.base import Strategy


# --- Step 1: Define a Strategy ---

class MovingAverageCrossover(Strategy):
    """Simple moving average crossover strategy."""

    name = "MA_Crossover"

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        super().__init__(short_window=short_window, long_window=long_window)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on MA crossover."""
        close = data["close"]
        short_ma = close.rolling(window=self.short_window, min_periods=1).mean()
        long_ma = close.rolling(window=self.long_window, min_periods=1).mean()

        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0
        signals.loc[short_ma > long_ma, "signal"] = 1
        signals.loc[short_ma < long_ma, "signal"] = -1
        return signals


# --- Step 2: Define an Algorithm ---

class EqualWeightAlgorithm(Algorithm):
    """Equal-weight portfolio optimization."""

    name = "EqualWeight"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        """Assign equal weights to all assets."""
        n = returns.shape[1]
        return np.ones(n) / n


# --- Step 3: Generate sample data ---

def generate_sample_data() -> pd.DataFrame:
    """Create synthetic OHLCV data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")

    close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 252)))
    high = close * (1 + np.abs(np.random.normal(0, 0.01, 252)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, 252)))
    open_ = close * (1 + np.random.normal(0, 0.005, 252))
    volume = np.random.randint(1_000_000, 10_000_000, 252).astype(float)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def main() -> None:
    """Run the end-to-end example."""
    # Generate data
    data = generate_sample_data()
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Create and apply strategy
    strategy = MovingAverageCrossover(short_window=10, long_window=30)
    signals = strategy.generate_signals(data)
    print(f"\nSignal distribution:")
    print(signals["signal"].value_counts())

    # Run backtest
    backtester = Backtester(
        strategy=strategy,
        initial_capital=1_000_000,
        commission=0.001,
        slippage=0.0005,
    )
    result = backtester.run(data)

    print(f"\n--- Backtest Results ---")
    print(f"Total Return: {result.metrics['total_return']:.2%}")
    print(f"Annualized Return: {result.metrics['annualized_return']:.2%}")
    print(f"Annualized Volatility: {result.metrics['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio: {result.metrics['sortino_ratio']:.3f}")
    print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio: {result.metrics['calmar_ratio']:.3f}")

    # Portfolio construction with multiple assets
    print(f"\n--- Portfolio Construction ---")
    n_assets = 5
    asset_names = [f"asset_{i}" for i in range(n_assets)]

    # Generate multi-asset returns
    np.random.seed(123)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    returns = pd.DataFrame(
        np.random.normal(0.0005, 0.02, (252, n_assets)),
        index=dates,
        columns=asset_names,
    )

    # All assets are "buy" signals
    multi_signals = pd.DataFrame(1, index=dates, columns=asset_names)

    algo = EqualWeightAlgorithm()
    constructor = PortfolioConstructor(algorithm=algo, rebalance_frequency="monthly")
    weights = constructor.construct(multi_signals, returns)

    print(f"Portfolio weights (last rebalance):")
    last_weights = weights.iloc[-1]
    for asset, w in last_weights.items():
        print(f"  {asset}: {w:.2%}")
    print(f"  Total: {last_weights.sum():.2%}")


if __name__ == "__main__":
    main()
