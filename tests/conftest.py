"""Shared test fixtures for the quant_portfolio test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Generate sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    n_assets = 5
    asset_names = [f"asset_{i}" for i in range(n_assets)]

    # Generate random walk prices starting at 100
    returns = np.random.normal(0.0005, 0.02, size=(252, n_assets))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    return pd.DataFrame(prices, index=dates, columns=asset_names)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate sample OHLCV data for a single asset."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")

    close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, 252)))
    high = close * (1 + np.abs(np.random.normal(0, 0.01, 252)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, 252)))
    open_ = close * (1 + np.random.normal(0, 0.005, 252))
    volume = np.random.randint(1000000, 10000000, 252).astype(float)

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


@pytest.fixture
def sample_returns(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Generate sample return data from prices."""
    return sample_prices.pct_change().dropna()


@pytest.fixture
def sample_signals(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Generate sample trading signals."""
    np.random.seed(42)
    signals = np.random.choice([-1, 0, 1], size=len(sample_ohlcv), p=[0.2, 0.6, 0.2])
    return pd.DataFrame({"signal": signals}, index=sample_ohlcv.index)
