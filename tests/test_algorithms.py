"""Tests for all 100 algorithm implementations."""

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.algorithms.registry import list_algorithms, get_algorithm


@pytest.fixture
def sample_returns():
    """Generate sample return data for testing."""
    rng = np.random.default_rng(42)
    n_obs = 252
    n_assets = 5
    # Generate correlated returns
    mu = np.array([0.0005, 0.0003, 0.0004, 0.0002, 0.0006])
    cov = np.array([
        [0.0004, 0.0001, 0.0001, 0.00005, 0.0001],
        [0.0001, 0.0003, 0.00008, 0.00004, 0.00009],
        [0.0001, 0.00008, 0.00035, 0.00006, 0.0001],
        [0.00005, 0.00004, 0.00006, 0.00025, 0.00005],
        [0.0001, 0.00009, 0.0001, 0.00005, 0.00045],
    ])
    returns_data = rng.multivariate_normal(mu, cov, size=n_obs)
    dates = pd.date_range("2023-01-01", periods=n_obs, freq="B")
    columns = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D", "ASSET_E"]
    return pd.DataFrame(returns_data, index=dates, columns=columns)


def test_all_algorithms_registered():
    """Test that at least 100 algorithms are registered."""
    algos = list_algorithms()
    assert len(algos) >= 100, f"Only {len(algos)} algorithms registered, expected >= 100"


def test_algorithm_names_unique():
    """Test that all algorithm names are unique."""
    algos = list_algorithms()
    assert len(algos) == len(set(algos))


@pytest.mark.parametrize("algo_name", list_algorithms())
def test_algorithm_instantiation(algo_name):
    """Test that each algorithm can be instantiated."""
    algo_class = get_algorithm(algo_name)
    algo = algo_class()
    assert algo is not None
    assert hasattr(algo, "optimize")


@pytest.mark.parametrize("algo_name", list_algorithms())
def test_algorithm_optimize(algo_name, sample_returns):
    """Test that each algorithm produces valid weight vectors."""
    algo_class = get_algorithm(algo_name)
    algo = algo_class()
    weights = algo.optimize(sample_returns)

    # Check output type and shape
    assert isinstance(weights, np.ndarray), f"{algo_name}: output is not ndarray"
    assert weights.shape == (5,), f"{algo_name}: shape is {weights.shape}, expected (5,)"

    # Check weights are finite
    assert np.all(np.isfinite(weights)), f"{algo_name}: weights contain non-finite values"

    # Check weights are non-negative (long-only)
    assert np.all(weights >= -1e-6), f"{algo_name}: negative weights found: {weights}"

    # Check weights sum to approximately 1
    weight_sum = weights.sum()
    assert abs(weight_sum - 1.0) < 0.05, (
        f"{algo_name}: weights sum to {weight_sum}, expected ~1.0"
    )


def test_algorithm_categories():
    """Test algorithms exist in expected categories."""
    algos = list_algorithms()

    # Check some from each category
    optimization_algos = [
        "mean_variance", "min_variance", "max_sharpe", "risk_parity",
        "black_litterman", "hierarchical_risk_parity",
    ]
    risk_algos = [
        "var_historical", "var_parametric", "var_monte_carlo",
        "drawdown_control", "risk_budgeting",
    ]
    allocation_algos = [
        "strategic_allocation", "tactical_allocation",
        "momentum_allocation", "volatility_weighted",
    ]
    signal_algos = [
        "kalman_filter", "wavelet_transform", "fourier_analysis",
    ]
    statistical_algos = [
        "covariance_estimation", "pca_decomposition", "regime_detection",
    ]
    numerical_algos = [
        "gradient_descent", "genetic_algorithm", "particle_swarm",
    ]
    ml_algos = [
        "neural_network_optimizer", "reinforcement_learning",
        "clustering_allocation",
    ]

    all_expected = (optimization_algos + risk_algos + allocation_algos +
                    signal_algos + statistical_algos + numerical_algos + ml_algos)

    for name in all_expected:
        assert name in algos, f"Expected algorithm '{name}' not found"


def test_algorithm_docstrings():
    """Test that all algorithms have docstrings."""
    for algo_name in list_algorithms():
        algo_class = get_algorithm(algo_name)
        assert algo_class.__doc__ is not None, f"{algo_name}: missing docstring"
        assert len(algo_class.__doc__) > 20, f"{algo_name}: docstring too short"
