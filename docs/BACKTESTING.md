# Backtesting Guide

This guide explains how to use the backtesting engine to evaluate trading strategies.

## Overview

The backtesting engine simulates strategy execution on historical (or synthetic) data, accounting for transaction costs and slippage, then computes comprehensive performance metrics.

## Basic Usage

```python
import numpy as np
import pandas as pd
from quant_portfolio.strategies.registry import get_strategy
from quant_portfolio.backtesting.engine import Backtester

# Generate sample data
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=252, freq="B")
close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 252)))
data = pd.DataFrame({
    "open": close * (1 + np.random.normal(0, 0.005, 252)),
    "high": close * (1 + np.abs(np.random.normal(0, 0.01, 252))),
    "low": close * (1 - np.abs(np.random.normal(0, 0.01, 252))),
    "close": close,
    "volume": np.random.randint(1_000_000, 10_000_000, 252).astype(float),
}, index=dates)

# Create a strategy
StrategyClass = get_strategy("simple_momentum")
strategy = StrategyClass()

# Configure and run the backtest
backtester = Backtester(
    strategy=strategy,
    initial_capital=1_000_000,
    commission=0.001,
    slippage=0.0005,
)
result = backtester.run(data)
```

## Backtester Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy` | Required | Strategy instance to backtest |
| `initial_capital` | 1,000,000 | Starting capital |
| `commission` | 0.001 | Commission rate per trade (0.1%) |
| `slippage` | 0.0005 | Slippage per trade (0.05%) |

## BacktestResult Object

The `run()` method returns a `BacktestResult` with the following attributes:

```python
result.returns        # pd.Series of daily strategy returns
result.equity_curve   # pd.Series of portfolio value over time
result.positions      # pd.DataFrame of position history
result.trades         # pd.DataFrame of trade log
result.metrics        # Dict of performance metrics
```

## Performance Metrics

The following metrics are computed automatically:

| Metric | Description |
|--------|-------------|
| `total_return` | Cumulative return over the period |
| `annualized_return` | Annualized compound return |
| `annualized_volatility` | Annualized standard deviation of returns |
| `sharpe_ratio` | Risk-adjusted return (excess return / volatility) |
| `sortino_ratio` | Downside risk-adjusted return |
| `max_drawdown` | Largest peak-to-trough decline |
| `calmar_ratio` | Annualized return / max drawdown |
| `num_trades` | Total number of trading periods |

## Comparing Strategies

```python
from quant_portfolio.strategies.registry import get_strategy
from quant_portfolio.backtesting.engine import Backtester

strategies_to_test = ["simple_momentum", "bollinger_bands", "dual_momentum"]

results = {}
for name in strategies_to_test:
    StrategyClass = get_strategy(name)
    strategy = StrategyClass()
    backtester = Backtester(strategy=strategy, initial_capital=1_000_000)
    results[name] = backtester.run(data)

# Compare Sharpe ratios
for name, result in results.items():
    print(f"{name}: Sharpe={result.metrics['sharpe_ratio']:.3f}")
```

## Transaction Cost Sensitivity

```python
for commission in [0.0, 0.001, 0.003, 0.005]:
    backtester = Backtester(
        strategy=strategy,
        initial_capital=1_000_000,
        commission=commission,
    )
    result = backtester.run(data)
    print(f"Commission={commission:.3f}: Return={result.metrics['total_return']:.2%}")
```

## Custom Price Column

By default, the backtester uses the "close" column. You can specify a different column:

```python
result = backtester.run(data, price_col="open")  # Execute on open prices
```

## Signal Requirements

Strategies must return a DataFrame with a `signal` column containing:
- `1` for buy/long signals
- `-1` for sell/short signals
- `0` for hold/neutral signals

Signals are applied with a one-bar delay (trade on the next bar after signal generation) to avoid look-ahead bias.
