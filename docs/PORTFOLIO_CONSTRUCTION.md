# Portfolio Construction Guide

This guide covers the end-to-end portfolio construction workflow using the Quant Portfolio library.

## Overview

Portfolio construction combines strategy signals with optimization algorithms to produce final portfolio weights. The process follows these steps:

1. **Generate Signals** - Apply strategies to market data
2. **Compute Returns** - Calculate asset returns for optimization
3. **Optimize Weights** - Use an algorithm to determine allocations
4. **Rebalance** - Apply weights at specified intervals

## The PortfolioConstructor Class

```python
from quant_portfolio.algorithms.registry import get_algorithm
from quant_portfolio.portfolio.constructor import PortfolioConstructor

AlgoClass = get_algorithm("risk_parity")
algo = AlgoClass()

constructor = PortfolioConstructor(
    algorithm=algo,
    rebalance_frequency="monthly",  # 'daily', 'weekly', 'monthly'
)

weights = constructor.construct(signals_df, returns_df)
```

## Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `algorithm` | Any Algorithm instance | Optimization algorithm for weight calculation |
| `rebalance_frequency` | `'daily'`, `'weekly'`, `'monthly'` | How often to recalculate weights |

## Input Requirements

### Signals DataFrame
- Columns: asset names
- Values: signal strength (non-zero means active, zero means excluded)
- Index: DatetimeIndex

### Returns DataFrame
- Columns: asset names (must match signals)
- Values: periodic returns (e.g., daily)
- Index: DatetimeIndex

## End-to-End Workflow

```python
import numpy as np
import pandas as pd
from quant_portfolio.strategies.registry import get_strategy
from quant_portfolio.algorithms.registry import get_algorithm
from quant_portfolio.backtesting.engine import Backtester
from quant_portfolio.portfolio.constructor import PortfolioConstructor

# Step 1: Generate signals for multiple assets
np.random.seed(42)
n_assets = 5
dates = pd.date_range("2020-01-01", periods=252, freq="B")

# Generate synthetic data for each asset
signals_list = []
returns_list = []

for i in range(n_assets):
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 252)))
    data = pd.DataFrame({
        "open": close * (1 + np.random.normal(0, 0.005, 252)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.01, 252))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.01, 252))),
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, 252).astype(float),
    }, index=dates)

    # Apply strategy
    StrategyClass = get_strategy("simple_momentum")
    strategy = StrategyClass()
    sigs = strategy.generate_signals(data)
    signals_list.append(sigs["signal"].rename(f"asset_{i}"))
    returns_list.append(data["close"].pct_change().rename(f"asset_{i}"))

# Combine into DataFrames
signals_df = pd.concat(signals_list, axis=1).fillna(0)
returns_df = pd.concat(returns_list, axis=1).dropna()

# Step 2: Construct portfolio
AlgoClass = get_algorithm("risk_parity")
algo = AlgoClass()

constructor = PortfolioConstructor(
    algorithm=algo,
    rebalance_frequency="monthly",
)

weights = constructor.construct(signals_df, returns_df)
print("Final weights:")
print(weights.iloc[-1])
```

## Comparing Algorithms

```python
algos_to_compare = ["equal_weight", "risk_parity", "min_variance", "max_sharpe"]

for algo_name in algos_to_compare:
    AlgoClass = get_algorithm(algo_name)
    algo = AlgoClass()
    constructor = PortfolioConstructor(algorithm=algo, rebalance_frequency="monthly")
    weights = constructor.construct(signals_df, returns_df)

    # Calculate portfolio returns
    port_returns = (weights * returns_df).sum(axis=1)
    sharpe = port_returns.mean() / port_returns.std() * np.sqrt(252)
    print(f"{algo_name}: Sharpe={sharpe:.3f}, Max Weight={weights.iloc[-1].max():.2%}")
```

## Risk Analysis

```python
from quant_portfolio.portfolio.risk import (
    calculate_var,
    calculate_cvar,
    calculate_volatility,
    calculate_beta,
)

# Calculate portfolio returns
portfolio_returns = (weights * returns_df).sum(axis=1)

# Risk metrics
var_95 = calculate_var(portfolio_returns, confidence=0.95)
cvar_95 = calculate_cvar(portfolio_returns, confidence=0.95)
ann_vol = calculate_volatility(portfolio_returns, annualize=True)

print(f"95% VaR: {var_95:.4f}")
print(f"95% CVaR: {cvar_95:.4f}")
print(f"Annualized Volatility: {ann_vol:.2%}")
```

## Rebalancing Frequency Impact

```python
for freq in ["daily", "weekly", "monthly"]:
    constructor = PortfolioConstructor(algorithm=algo, rebalance_frequency=freq)
    weights = constructor.construct(signals_df, returns_df)
    port_returns = (weights * returns_df).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).mean()
    print(f"{freq}: Avg Daily Turnover={turnover:.4f}")
```

## Integration with Backtesting

After constructing a portfolio, you can analyze individual strategy performance:

```python
from quant_portfolio.backtesting.engine import Backtester

# Backtest each strategy individually
for i in range(n_assets):
    StrategyClass = get_strategy("simple_momentum")
    strategy = StrategyClass()
    backtester = Backtester(strategy=strategy, initial_capital=1_000_000)
    result = backtester.run(asset_data[i])
    print(f"Asset {i}: Sharpe={result.metrics['sharpe_ratio']:.3f}")
```
