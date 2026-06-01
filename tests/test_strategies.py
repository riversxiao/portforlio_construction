"""Tests for all 100 quantitative trading strategies."""

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.strategies.registry import list_strategies, get_strategy


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for strategy testing."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10)  # Keep positive

    data = pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.2,
            "high": close + np.abs(np.random.randn(n) * 0.5),
            "low": close - np.abs(np.random.randn(n) * 0.5),
            "close": close,
            "volume": np.random.randint(100000, 1000000, n).astype(float),
        },
        index=dates,
    )
    # Ensure high >= close and low <= close
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)
    return data


def test_strategy_count():
    """Test that at least 100 strategies are registered."""
    strategies = list_strategies()
    assert len(strategies) >= 100, (
        f"Expected >= 100 strategies, got {len(strategies)}: {strategies}"
    )


def test_all_strategies_instantiate(sample_ohlcv_data):
    """Test that all strategies can be instantiated."""
    strategies = list_strategies()
    for name in strategies:
        cls = get_strategy(name)
        instance = cls()
        assert instance is not None, f"Failed to instantiate {name}"


def test_all_strategies_generate_signals(sample_ohlcv_data):
    """Test that all strategies generate valid signal DataFrames."""
    strategies = list_strategies()
    for name in strategies:
        cls = get_strategy(name)
        instance = cls()
        result = instance.generate_signals(sample_ohlcv_data)

        assert isinstance(result, pd.DataFrame), (
            f"{name}: Expected DataFrame, got {type(result)}"
        )
        assert "signal" in result.columns, (
            f"{name}: Missing 'signal' column"
        )
        assert len(result) == len(sample_ohlcv_data), (
            f"{name}: Length mismatch"
        )
        # Signal values should be in {-1, 0, 1}
        unique_vals = set(result["signal"].dropna().unique())
        valid_vals = {-1, 0, 1, -1.0, 0.0, 1.0}
        assert unique_vals.issubset(valid_vals), (
            f"{name}: Invalid signal values {unique_vals - valid_vals}"
        )


def test_strategy_has_docstring(sample_ohlcv_data):
    """Test that all strategies have docstrings."""
    strategies = list_strategies()
    for name in strategies:
        cls = get_strategy(name)
        assert cls.__doc__ is not None, f"{name}: Missing class docstring"
        assert cls.generate_signals.__doc__ is not None, (
            f"{name}: Missing generate_signals docstring"
        )


def test_strategy_names_unique():
    """Test that all strategy names are unique."""
    strategies = list_strategies()
    assert len(strategies) == len(set(strategies)), "Duplicate strategy names found"


def test_get_strategy_by_name():
    """Test get_strategy returns correct class."""
    cls = get_strategy("simple_momentum")
    assert cls.name == "simple_momentum"
    instance = cls()
    assert hasattr(instance, "generate_signals")


def test_get_strategy_not_found():
    """Test get_strategy raises for unknown strategy."""
    with pytest.raises(KeyError):
        get_strategy("nonexistent_strategy")
