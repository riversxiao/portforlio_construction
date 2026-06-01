# Contributing to Quant Portfolio

Thank you for your interest in contributing to the Quant Portfolio library! This guide explains how to add new strategies, algorithms, and other improvements.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/portforlio_construction.git
cd portforlio_construction

# Install in development mode with test dependencies
pip install -e '.[dev]'

# Run tests to verify setup
pytest tests/ -v
```

## Adding a New Strategy

### 1. Choose the Category

Place your strategy in the appropriate category directory under `src/quant_portfolio/strategies/`:

- `momentum/` - Price momentum and relative strength
- `mean_reversion/` - Mean reversion and pairs trading
- `trend_following/` - Trend detection and following
- `volatility/` - Volatility-based trading
- `stat_arb/` - Statistical arbitrage
- `factor/` - Factor investing
- `ml/` - Machine learning strategies
- `technical/` - Technical indicator strategies

### 2. Create the Strategy File

Create a new Python file in the chosen category directory:

```python
"""Description of your strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_portfolio.strategies.base import Strategy


class MyNewStrategy(Strategy):
    """Short description of the strategy.

    Longer description explaining the mathematical basis,
    when it works best, and any caveats.

    Parameters
    ----------
    lookback : int
        Lookback period for signal calculation.
    threshold : float
        Signal generation threshold.
    """

    name = "my_new_strategy"  # Unique snake_case identifier

    def __init__(self, lookback: int = 20, threshold: float = 1.5) -> None:
        super().__init__(lookback=lookback, threshold=threshold)
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from OHLCV data.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'signal' column: 1 (buy), -1 (sell), 0 (hold).
        """
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0

        # Your signal logic here
        close = data["close"]
        # Example: simple z-score approach
        rolling_mean = close.rolling(self.lookback, min_periods=1).mean()
        rolling_std = close.rolling(self.lookback, min_periods=1).std().fillna(1)
        zscore = (close - rolling_mean) / rolling_std

        signals.loc[zscore > self.threshold, "signal"] = -1
        signals.loc[zscore < -self.threshold, "signal"] = 1

        return signals
```

### 3. Register the Strategy

The strategy is auto-registered via `__init_subclass__`. Just make sure it is imported in the category's `__init__.py`:

```python
# In src/quant_portfolio/strategies/mean_reversion/__init__.py
from quant_portfolio.strategies.mean_reversion.my_new_strategy import MyNewStrategy
```

### 4. Verify Registration

```python
from quant_portfolio.strategies.registry import get_strategy, list_strategies

assert "my_new_strategy" in list_strategies()
StrategyClass = get_strategy("my_new_strategy")
strategy = StrategyClass()
```

## Adding a New Algorithm

### 1. Choose the Category

Place your algorithm in the appropriate directory under `src/quant_portfolio/algorithms/`:

- `optimization/` - Portfolio optimization (mean-variance, risk parity, etc.)
- `risk/` - Risk management and measurement
- `allocation/` - Asset allocation methods
- `signal_processing/` - Signal filtering and denoising
- `statistical/` - Statistical estimation and testing
- `numerical/` - Numerical optimization methods
- `ml_optimization/` - Machine learning approaches

### 2. Create the Algorithm File

```python
"""Description of your algorithm."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_portfolio.algorithms.base import Algorithm


class MyNewAlgorithm(Algorithm):
    """Short description of the algorithm.

    Mathematical basis:
        Describe the optimization objective and constraints.

    Parameters
    ----------
    param1 : float
        Description of parameter.
    """

    name = "my_new_algorithm"  # Unique snake_case identifier

    def __init__(self, param1: float = 0.5, **kwargs: Any) -> None:
        super().__init__(param1=param1, **kwargs)
        self.param1 = param1

    def optimize(self, returns: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Compute optimal portfolio weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns DataFrame (rows=periods, columns=assets).

        Returns
        -------
        np.ndarray
            Portfolio weights summing to approximately 1.0.
        """
        n_assets = returns.shape[1]

        # Your optimization logic here
        # Example: volatility-inverse weighting
        vols = returns.std().values
        inv_vols = 1.0 / np.maximum(vols, 1e-8)
        weights = inv_vols / inv_vols.sum()

        return weights
```

### 3. Register the Algorithm

Add the import to the category's `__init__.py`:

```python
# In src/quant_portfolio/algorithms/optimization/__init__.py
from quant_portfolio.algorithms.optimization.my_new_algorithm import MyNewAlgorithm
```

### 4. Verify Registration

```python
from quant_portfolio.algorithms.registry import get_algorithm, list_algorithms

assert "my_new_algorithm" in list_algorithms()
AlgoClass = get_algorithm("my_new_algorithm")
algo = AlgoClass()
```

## Requirements for Contributions

### Code Quality

- All code must include type hints
- Every class and public method needs a docstring (NumPy style)
- Use `from __future__ import annotations` at the top of every module
- Follow the existing code style (formatting, naming conventions)

### Strategy Requirements

- Must extend `Strategy` base class
- Must implement `generate_signals(data: pd.DataFrame) -> pd.DataFrame`
- Signal DataFrame must have a `signal` column with values in {-1, 0, 1}
- Must handle edge cases (short data, NaN values) gracefully
- Must work on standard OHLCV data (columns: open, high, low, close, volume)

### Algorithm Requirements

- Must extend `Algorithm` base class
- Must implement `optimize(returns: pd.DataFrame) -> np.ndarray`
- Output weights must sum to approximately 1.0
- Must handle degenerate cases (constant returns, single asset, etc.)
- Should not require external API calls or network access

### Testing

- Add tests that verify your strategy/algorithm works on synthetic data
- Run the full test suite before submitting: `pytest tests/ -v`
- Test edge cases: short data, constant prices, all-zero volumes

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_integration.py -v

# Run with coverage
pytest tests/ --cov=quant_portfolio --cov-report=term-missing
```

### Dependencies

- Prefer numpy and scipy for numerical computation
- Avoid adding new dependencies unless absolutely necessary
- If a new dependency is needed, add it to `pyproject.toml`

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-strategy`
3. Make your changes following the guidelines above
4. Run the test suite: `pytest tests/ -v`
5. Commit with a descriptive message: `feat: add my_new_strategy`
6. Push and open a Pull Request

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
