"""Tests for the portfolio constructor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.algorithms.base import Algorithm, AlgorithmRegistry
from quant_portfolio.portfolio.constructor import PortfolioConstructor
from quant_portfolio.portfolio.risk import (
    calculate_beta,
    calculate_cvar,
    calculate_tracking_error,
    calculate_var,
    calculate_volatility,
)
from quant_portfolio.strategies.base import Strategy, StrategyRegistry


class EqualWeightAlgorithm(Algorithm):
    """Simple equal-weight algorithm for testing."""

    name = "EqualWeight"

    def optimize(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        """Return equal weights for all assets."""
        n_assets = returns.shape[1]
        return np.ones(n_assets) / n_assets


class TestPortfolioConstructor:
    """Tests for the PortfolioConstructor class."""

    def test_constructor_instantiation(self) -> None:
        """Test that constructor can be instantiated."""
        algo = EqualWeightAlgorithm()
        constructor = PortfolioConstructor(algorithm=algo)
        assert constructor.rebalance_frequency == "monthly"

    def test_construct_weights(self, sample_returns: pd.DataFrame) -> None:
        """Test that construction produces valid weights."""
        algo = EqualWeightAlgorithm()
        constructor = PortfolioConstructor(
            algorithm=algo, rebalance_frequency="monthly"
        )

        # Create signals (all buy)
        signals = pd.DataFrame(
            1, index=sample_returns.index, columns=sample_returns.columns
        )

        weights = constructor.construct(signals, sample_returns)

        assert isinstance(weights, pd.DataFrame)
        assert weights.shape[1] == sample_returns.shape[1]
        # Weights should sum to approximately 1 for rows with active positions
        non_zero_rows = weights[weights.sum(axis=1) > 0]
        if len(non_zero_rows) > 0:
            weight_sums = non_zero_rows.sum(axis=1)
            np.testing.assert_allclose(weight_sums, 1.0, atol=1e-6)

    def test_construct_with_no_signals(self, sample_returns: pd.DataFrame) -> None:
        """Test construction with zero signals produces zero weights."""
        algo = EqualWeightAlgorithm()
        constructor = PortfolioConstructor(
            algorithm=algo, rebalance_frequency="monthly"
        )

        # Create signals (all hold/no position)
        signals = pd.DataFrame(
            0, index=sample_returns.index, columns=sample_returns.columns
        )

        weights = constructor.construct(signals, sample_returns)
        assert (weights == 0).all().all()


class TestAlgorithmRegistry:
    """Tests for the AlgorithmRegistry."""

    def test_algorithm_auto_registration(self) -> None:
        """Test that concrete algorithms are auto-registered."""
        assert "EqualWeight" in AlgorithmRegistry.list_algorithms()

    def test_get_algorithm(self) -> None:
        """Test getting a registered algorithm."""
        algo_cls = AlgorithmRegistry.get("EqualWeight")
        assert algo_cls == EqualWeightAlgorithm

    def test_get_nonexistent_algorithm(self) -> None:
        """Test that getting a nonexistent algorithm raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            AlgorithmRegistry.get("NonexistentAlgorithm")


class TestStrategyRegistry:
    """Tests for the StrategyRegistry."""

    def test_strategy_interface(self) -> None:
        """Test that Strategy is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            Strategy()  # type: ignore

    def test_algorithm_interface(self) -> None:
        """Test that Algorithm is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            Algorithm()  # type: ignore


class TestRiskUtilities:
    """Tests for risk analysis utilities."""

    def test_calculate_var(self) -> None:
        """Test VaR calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 252))
        var = calculate_var(returns, confidence=0.95)
        assert var > 0
        assert isinstance(var, float)

    def test_calculate_cvar(self) -> None:
        """Test CVaR calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 252))
        cvar = calculate_cvar(returns, confidence=0.95)
        var = calculate_var(returns, confidence=0.95)
        # CVaR should be >= VaR
        assert cvar >= var

    def test_calculate_volatility(self) -> None:
        """Test volatility calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 252))
        vol = calculate_volatility(returns)
        assert vol > 0

    def test_calculate_beta(self) -> None:
        """Test beta calculation."""
        np.random.seed(42)
        benchmark = pd.Series(np.random.normal(0.001, 0.01, 252))
        # Portfolio correlated with benchmark
        portfolio = benchmark * 1.2 + np.random.normal(0, 0.005, 252)
        beta = calculate_beta(
            pd.Series(portfolio, index=range(252)),
            pd.Series(benchmark.values, index=range(252)),
        )
        assert isinstance(beta, float)
        # Beta should be roughly 1.2
        assert 0.5 < beta < 2.0

    def test_calculate_tracking_error(self) -> None:
        """Test tracking error calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        benchmark = pd.Series(np.random.normal(0.0008, 0.015, 252))
        te = calculate_tracking_error(returns, benchmark)
        assert te > 0
