"""Tests for the backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.backtesting.engine import Backtester, BacktestResult
from quant_portfolio.backtesting.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from quant_portfolio.strategies.base import Strategy


class SimpleTestStrategy(Strategy):
    """A simple strategy for testing that buys when price is above MA."""

    name = "SimpleTestStrategy"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on simple moving average crossover."""
        close = data["close"]
        ma = close.rolling(window=20, min_periods=1).mean()
        signals = pd.DataFrame(index=data.index)
        signals["signal"] = 0
        signals.loc[close > ma, "signal"] = 1
        signals.loc[close < ma, "signal"] = -1
        return signals


class TestBacktester:
    """Tests for the Backtester class."""

    def test_backtester_instantiation(self) -> None:
        """Test that Backtester can be instantiated."""
        strategy = SimpleTestStrategy()
        bt = Backtester(strategy=strategy)
        assert bt.initial_capital == 1_000_000.0
        assert bt.commission == 0.001
        assert bt.slippage == 0.0005

    def test_backtester_run(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test running a backtest produces valid results."""
        strategy = SimpleTestStrategy()
        bt = Backtester(strategy=strategy)
        result = bt.run(sample_ohlcv)

        assert isinstance(result, BacktestResult)
        assert isinstance(result.returns, pd.Series)
        assert isinstance(result.equity_curve, pd.Series)
        assert isinstance(result.metrics, dict)
        assert len(result.returns) == len(sample_ohlcv)

    def test_backtester_metrics_present(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test that all expected metrics are in the result."""
        strategy = SimpleTestStrategy()
        bt = Backtester(strategy=strategy)
        result = bt.run(sample_ohlcv)

        expected_metrics = [
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "total_return",
        ]
        for metric in expected_metrics:
            assert metric in result.metrics

    def test_backtester_custom_params(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test backtest with custom parameters."""
        strategy = SimpleTestStrategy()
        bt = Backtester(
            strategy=strategy,
            initial_capital=500_000.0,
            commission=0.002,
            slippage=0.001,
        )
        result = bt.run(sample_ohlcv)

        assert result.equity_curve.iloc[0] != 0
        assert isinstance(result.metrics["sharpe_ratio"], float)

    def test_backtester_equity_curve_starts_near_capital(
        self, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Test that equity curve starts near initial capital."""
        strategy = SimpleTestStrategy()
        bt = Backtester(strategy=strategy, initial_capital=1_000_000.0)
        result = bt.run(sample_ohlcv)

        # First value should be close to initial capital
        assert abs(result.equity_curve.iloc[0] - 1_000_000.0) < 50_000


class TestMetrics:
    """Tests for standalone metrics functions."""

    def test_annualized_return(self) -> None:
        """Test annualized return calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        ann_ret = annualized_return(returns)
        assert isinstance(ann_ret, float)

    def test_sharpe_ratio(self) -> None:
        """Test Sharpe ratio calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        sr = sharpe_ratio(returns)
        assert isinstance(sr, float)

    def test_max_drawdown_is_negative(self) -> None:
        """Test that max drawdown is negative or zero."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        mdd = max_drawdown(returns)
        assert mdd <= 0

    def test_annualized_volatility(self) -> None:
        """Test annualized volatility calculation."""
        returns = pd.Series(np.random.normal(0, 0.02, 252))
        vol = annualized_volatility(returns)
        assert vol > 0

    def test_sortino_ratio(self) -> None:
        """Test Sortino ratio calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        sr = sortino_ratio(returns)
        assert isinstance(sr, float)
