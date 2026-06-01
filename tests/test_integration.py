"""End-to-end integration tests for the full pipeline.

Tests the complete workflow:
1. Generate synthetic OHLCV data
2. Apply strategies from different categories
3. Run backtest on each
4. Optimize with different algorithms
5. Construct final portfolio
6. Verify metrics are computed correctly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.algorithms.registry import get_algorithm, list_algorithms
from quant_portfolio.backtesting.engine import Backtester
from quant_portfolio.portfolio.constructor import PortfolioConstructor
from quant_portfolio.portfolio.risk import (
    calculate_beta,
    calculate_cvar,
    calculate_tracking_error,
    calculate_var,
    calculate_volatility,
)
from quant_portfolio.strategies.registry import get_strategy, list_strategies


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Generate synthetic OHLCV data for integration testing."""
    np.random.seed(42)
    n_periods = 300
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="B")

    close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n_periods)))
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
    open_ = close * (1 + np.random.normal(0, 0.005, n_periods))
    volume = np.random.randint(1_000_000, 10_000_000, n_periods).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def multi_asset_returns() -> pd.DataFrame:
    """Generate synthetic multi-asset return data."""
    np.random.seed(123)
    n_periods = 300
    n_assets = 5
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="B")
    asset_names = [f"asset_{i}" for i in range(n_assets)]

    # Factor model for correlated returns
    factor = np.random.normal(0.0003, 0.01, n_periods)
    betas = np.array([0.8, 1.0, 1.2, 0.6, 1.4])
    idio = np.random.normal(0, 0.008, (n_periods, n_assets))
    returns = np.outer(factor, betas) + idio + np.array([0.0002, 0.0003, 0.0004, 0.0001, 0.0005])

    return pd.DataFrame(returns, index=dates, columns=asset_names)


@pytest.fixture
def multi_asset_ohlcv_list() -> list[pd.DataFrame]:
    """Generate OHLCV data for multiple assets."""
    np.random.seed(77)
    n_periods = 300
    n_assets = 5
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="B")

    datasets = []
    for i in range(n_assets):
        drift = 0.0003 + i * 0.0001
        vol = 0.015 + i * 0.003
        close = 100 * np.exp(np.cumsum(np.random.normal(drift, vol, n_periods)))
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
        open_ = close * (1 + np.random.normal(0, 0.005, n_periods))
        volume = np.random.randint(1_000_000, 10_000_000, n_periods).astype(float)
        datasets.append(
            pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                index=dates,
            )
        )
    return datasets


class TestStrategyIntegration:
    """Integration tests for strategies from different categories."""

    @pytest.mark.parametrize(
        "strategy_name",
        [
            "simple_momentum",       # momentum
            "bollinger_bands",       # mean_reversion
            "moving_average_crossover",  # trend_following
            "garch_vol",             # volatility
            "accumulation_distribution",  # technical
        ],
    )
    def test_strategy_generates_valid_signals(
        self, strategy_name: str, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        """Strategies from different categories produce valid signals."""
        StrategyClass = get_strategy(strategy_name)
        strategy = StrategyClass()
        signals = strategy.generate_signals(synthetic_ohlcv)

        assert "signal" in signals.columns
        assert len(signals) == len(synthetic_ohlcv)
        assert signals["signal"].isin([-1, 0, 1]).all()
        # At least some non-zero signals
        assert (signals["signal"] != 0).any()

    def test_all_strategies_registered(self) -> None:
        """Verify all 100 strategies are registered."""
        strategies = list_strategies()
        assert len(strategies) >= 100


class TestBacktestIntegration:
    """Integration tests for the backtesting engine."""

    @pytest.mark.parametrize(
        "strategy_name",
        [
            "simple_momentum",
            "bollinger_bands",
            "moving_average_crossover",
            "garch_vol",
            "accumulation_distribution",
        ],
    )
    def test_backtest_produces_valid_results(
        self, strategy_name: str, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        """Backtest each strategy and verify result structure and metrics."""
        StrategyClass = get_strategy(strategy_name)
        strategy = StrategyClass()

        backtester = Backtester(
            strategy=strategy,
            initial_capital=1_000_000,
            commission=0.001,
            slippage=0.0005,
        )
        result = backtester.run(synthetic_ohlcv)

        # Verify result structure
        assert result.returns is not None
        assert result.equity_curve is not None
        assert result.positions is not None
        assert result.metrics is not None

        # Verify metric keys exist
        expected_keys = [
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
        ]
        for key in expected_keys:
            assert key in result.metrics, f"Missing metric: {key}"

        # Verify metrics are finite
        for key, value in result.metrics.items():
            assert np.isfinite(value), f"Non-finite metric {key}: {value}"

        # Equity curve starts at initial capital
        assert abs(result.equity_curve.iloc[0] - 1_000_000) < 1_000_000 * 0.05

        # Max drawdown should be non-positive (or zero)
        assert result.metrics["max_drawdown"] <= 0.0


class TestAlgorithmIntegration:
    """Integration tests for optimization algorithms."""

    @pytest.mark.parametrize(
        "algo_name",
        [
            "equal_weight",          # optimization
            "risk_parity",           # optimization
            "var_historical",        # risk
            "momentum_allocation",   # allocation
            "covariance_estimation", # statistical
        ],
    )
    def test_algorithm_produces_valid_weights(
        self, algo_name: str, multi_asset_returns: pd.DataFrame
    ) -> None:
        """Algorithms from different categories produce valid weight vectors."""
        AlgoClass = get_algorithm(algo_name)
        algo = AlgoClass()
        weights = algo.optimize(multi_asset_returns)

        assert isinstance(weights, np.ndarray)
        assert len(weights) == multi_asset_returns.shape[1]
        # Weights should sum to approximately 1
        assert abs(weights.sum() - 1.0) < 0.1, (
            f"{algo_name} weights sum to {weights.sum():.4f}"
        )
        # All weights should be finite
        assert np.all(np.isfinite(weights))

    def test_all_algorithms_registered(self) -> None:
        """Verify all 100 algorithms are registered."""
        algorithms = list_algorithms()
        assert len(algorithms) >= 100


class TestPortfolioConstructionIntegration:
    """Integration tests for the full portfolio construction pipeline."""

    def test_full_pipeline(
        self,
        multi_asset_ohlcv_list: list[pd.DataFrame],
        multi_asset_returns: pd.DataFrame,
    ) -> None:
        """Test complete pipeline: signals -> backtest -> optimize -> construct."""
        n_assets = 5
        asset_names = [f"asset_{i}" for i in range(n_assets)]
        dates = multi_asset_returns.index

        # Step 1: Generate signals for each asset
        strategies_to_use = [
            "simple_momentum",
            "bollinger_bands",
            "moving_average_crossover",
            "rsi_momentum",
            "macd_momentum",
        ]

        signals_list = []
        for i, strat_name in enumerate(strategies_to_use):
            StrategyClass = get_strategy(strat_name)
            strategy = StrategyClass()
            sigs = strategy.generate_signals(multi_asset_ohlcv_list[i])
            signals_list.append(sigs["signal"].rename(asset_names[i]))

        signals_df = pd.concat(signals_list, axis=1).reindex(dates).fillna(0)
        assert signals_df.shape == (len(dates), n_assets)

        # Step 2: Backtest each strategy individually
        for i, strat_name in enumerate(strategies_to_use):
            StrategyClass = get_strategy(strat_name)
            strategy = StrategyClass()
            backtester = Backtester(strategy=strategy, initial_capital=1_000_000)
            result = backtester.run(multi_asset_ohlcv_list[i])
            assert result.metrics["total_return"] is not None

        # Step 3: Optimize with algorithm
        AlgoClass = get_algorithm("risk_parity")
        algo = AlgoClass()

        # Step 4: Construct portfolio
        constructor = PortfolioConstructor(
            algorithm=algo,
            rebalance_frequency="monthly",
        )
        weights = constructor.construct(signals_df, multi_asset_returns)

        # Verify weights DataFrame
        assert weights.shape[1] == n_assets
        assert len(weights) == len(multi_asset_returns)

        # Weights should be non-negative and bounded
        assert (weights >= -0.01).all().all()
        # On non-zero rows, weights should sum to approximately 1
        non_zero_rows = weights[weights.sum(axis=1) > 0.01]
        if len(non_zero_rows) > 0:
            sums = non_zero_rows.sum(axis=1)
            assert (sums > 0.5).all(), f"Some weight sums too low: {sums.min()}"
            assert (sums < 1.5).all(), f"Some weight sums too high: {sums.max()}"

    def test_portfolio_construction_with_multiple_algos(
        self,
        multi_asset_ohlcv_list: list[pd.DataFrame],
        multi_asset_returns: pd.DataFrame,
    ) -> None:
        """Test that different algorithms produce different but valid portfolios."""
        n_assets = 5
        asset_names = [f"asset_{i}" for i in range(n_assets)]
        dates = multi_asset_returns.index

        # Generate signals using a single strategy for all assets
        StrategyClass = get_strategy("simple_momentum")
        signals_list = []
        for i in range(n_assets):
            strategy = StrategyClass()
            sigs = strategy.generate_signals(multi_asset_ohlcv_list[i])
            signals_list.append(sigs["signal"].rename(asset_names[i]))
        signals_df = pd.concat(signals_list, axis=1).reindex(dates).fillna(0)

        # Test with 5 different algorithms
        algos_to_test = [
            "equal_weight",
            "min_variance",
            "risk_parity",
            "inverse_variance",
            "max_diversification",
        ]

        results = {}
        for algo_name in algos_to_test:
            AlgoClass = get_algorithm(algo_name)
            algo = AlgoClass()
            constructor = PortfolioConstructor(
                algorithm=algo, rebalance_frequency="monthly"
            )
            weights = constructor.construct(signals_df, multi_asset_returns)
            port_returns = (weights * multi_asset_returns).sum(axis=1)
            results[algo_name] = {
                "weights": weights,
                "returns": port_returns,
                "sharpe": (
                    port_returns.mean() / port_returns.std() * np.sqrt(252)
                    if port_returns.std() > 0
                    else 0.0
                ),
            }

        # Verify all produced results
        for algo_name, res in results.items():
            assert res["weights"] is not None
            assert len(res["returns"]) > 0
            assert np.isfinite(res["sharpe"]), f"{algo_name} has non-finite Sharpe"


class TestRiskMetricsIntegration:
    """Integration tests for risk metrics on portfolio returns."""

    def test_risk_metrics_on_constructed_portfolio(
        self,
        multi_asset_ohlcv_list: list[pd.DataFrame],
        multi_asset_returns: pd.DataFrame,
    ) -> None:
        """Verify risk metrics work correctly on constructed portfolio returns."""
        n_assets = 5
        asset_names = [f"asset_{i}" for i in range(n_assets)]
        dates = multi_asset_returns.index

        # Build a simple portfolio
        StrategyClass = get_strategy("simple_momentum")
        signals_list = []
        for i in range(n_assets):
            strategy = StrategyClass()
            sigs = strategy.generate_signals(multi_asset_ohlcv_list[i])
            signals_list.append(sigs["signal"].rename(asset_names[i]))
        signals_df = pd.concat(signals_list, axis=1).reindex(dates).fillna(0)

        AlgoClass = get_algorithm("equal_weight")
        algo = AlgoClass()
        constructor = PortfolioConstructor(algorithm=algo, rebalance_frequency="monthly")
        weights = constructor.construct(signals_df, multi_asset_returns)
        portfolio_returns = (weights * multi_asset_returns).sum(axis=1)

        # Test VaR
        var_95 = calculate_var(portfolio_returns, confidence=0.95, method="historical")
        assert var_95 >= 0, "VaR should be non-negative"
        assert np.isfinite(var_95)

        # Test CVaR
        cvar_95 = calculate_cvar(portfolio_returns, confidence=0.95)
        assert cvar_95 >= var_95 or abs(cvar_95 - var_95) < 0.001, (
            "CVaR should be >= VaR"
        )
        assert np.isfinite(cvar_95)

        # Test volatility
        vol = calculate_volatility(portfolio_returns, annualize=True)
        assert vol > 0, "Volatility should be positive"
        assert vol < 2.0, "Volatility unreasonably high"
        assert np.isfinite(vol)

        # Test beta
        benchmark = multi_asset_returns.mean(axis=1)  # equal-weight benchmark
        beta = calculate_beta(portfolio_returns, benchmark)
        assert np.isfinite(beta)
        assert abs(beta) < 5.0, "Beta unreasonably large"

        # Test tracking error
        te = calculate_tracking_error(portfolio_returns, benchmark, annualize=True)
        assert te >= 0, "Tracking error should be non-negative"
        assert np.isfinite(te)

    def test_var_parametric_vs_historical(self) -> None:
        """Verify VaR methods produce consistent results."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 500))

        var_hist = calculate_var(returns, confidence=0.95, method="historical")
        var_param = calculate_var(returns, confidence=0.95, method="parametric")

        # Both should be positive
        assert var_hist > 0
        assert var_param > 0

        # They should be in the same ballpark (within 50% of each other)
        ratio = var_hist / var_param if var_param > 0 else float("inf")
        assert 0.5 < ratio < 2.0, f"VaR methods diverge too much: {ratio:.2f}"


class TestEndToEndPipeline:
    """Full end-to-end integration test combining everything."""

    def test_complete_workflow(self, synthetic_ohlcv: pd.DataFrame) -> None:
        """Run the complete workflow from data to final portfolio metrics."""
        # Step 1: Apply strategy
        StrategyClass = get_strategy("dual_momentum")
        strategy = StrategyClass()
        signals = strategy.generate_signals(synthetic_ohlcv)
        assert "signal" in signals.columns

        # Step 2: Backtest
        backtester = Backtester(
            strategy=strategy,
            initial_capital=1_000_000,
            commission=0.001,
            slippage=0.0005,
        )
        result = backtester.run(synthetic_ohlcv)
        assert result.metrics["sharpe_ratio"] is not None
        assert np.isfinite(result.metrics["total_return"])

        # Step 3: Multi-asset optimization
        np.random.seed(88)
        n_assets = 4
        dates = synthetic_ohlcv.index
        multi_returns = pd.DataFrame(
            np.random.normal(0.0004, 0.018, (len(dates), n_assets)),
            index=dates,
            columns=[f"s_{i}" for i in range(n_assets)],
        )

        AlgoClass = get_algorithm("risk_parity")
        algo = AlgoClass()
        weights = algo.optimize(multi_returns)
        assert abs(weights.sum() - 1.0) < 0.05
        assert np.all(weights >= -0.01)

        # Step 4: Portfolio construction
        multi_signals = pd.DataFrame(
            np.where(np.random.random((len(dates), n_assets)) > 0.3, 1, 0),
            index=dates,
            columns=[f"s_{i}" for i in range(n_assets)],
        )

        constructor = PortfolioConstructor(
            algorithm=algo,
            rebalance_frequency="monthly",
        )
        port_weights = constructor.construct(multi_signals, multi_returns)
        assert port_weights.shape == (len(dates), n_assets)

        # Step 5: Compute risk metrics
        port_returns = (port_weights * multi_returns).sum(axis=1)
        vol = calculate_volatility(port_returns, annualize=True)
        var = calculate_var(port_returns, confidence=0.95)

        assert np.isfinite(vol)
        assert np.isfinite(var)
        assert vol > 0
