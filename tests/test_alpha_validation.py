"""Comprehensive tests for the Alpha Validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.backtesting.engine import BacktestResult
from quant_portfolio.validation import (
    AlphaValidator,
    DimensionResult,
    ValidationConfig,
    ValidationReport,
)
from quant_portfolio.validation.dimensions import (
    CoverageEvaluator,
    DrawdownRiskEvaluator,
    EfficiencyEvaluator,
    IndependenceEvaluator,
    ReturnAbilityEvaluator,
    RiskAdjustedEvaluator,
    RobustnessEvaluator,
    TransactionCostEvaluator,
)


def _make_backtest_result(
    returns: pd.Series,
    positions: pd.DataFrame | None = None,
) -> BacktestResult:
    """Helper to create a BacktestResult from returns and optional positions."""
    equity_curve = (1 + returns).cumprod()
    if positions is None:
        positions = pd.DataFrame(
            {"position": np.ones(len(returns))}, index=returns.index
        )
    trades = pd.DataFrame(columns=["date", "side", "quantity", "price"])
    metrics = {
        "total_return": float((1 + returns).prod() - 1),
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
    }
    return BacktestResult(
        returns=returns,
        equity_curve=equity_curve,
        positions=positions,
        trades=trades,
        metrics=metrics,
    )


class TestDimensionEvaluators:
    """Tests for each of the 8 dimension evaluators."""

    def test_return_ability_evaluator_basic(self) -> None:
        """Synthetic returns with known properties, verify metrics are computed."""
        np.random.seed(100)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)

        evaluator = ReturnAbilityEvaluator()
        result = evaluator.evaluate(returns)

        assert isinstance(result, DimensionResult)
        assert result.dimension_name == "return_ability"
        assert "total_return" in result.metrics
        assert "annualized_return" in result.metrics
        assert "hit_rate" in result.metrics
        assert "profit_factor" in result.metrics
        assert "avg_win_loss_ratio" in result.metrics
        assert 0 <= result.score <= 100

    def test_return_ability_high_returns(self) -> None:
        """Verify high-return series gets score > 70."""
        np.random.seed(101)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        # High positive bias returns
        returns = pd.Series(np.random.normal(0.005, 0.005, 252), index=dates)

        evaluator = ReturnAbilityEvaluator()
        result = evaluator.evaluate(returns)

        assert result.score > 70
        assert result.passed is True

    def test_risk_adjusted_evaluator_basic(self) -> None:
        """Verify sharpe/sortino/omega are computed."""
        np.random.seed(102)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)

        evaluator = RiskAdjustedEvaluator()
        result = evaluator.evaluate(returns)

        assert "sharpe_ratio" in result.metrics
        assert "sortino_ratio" in result.metrics
        assert "omega_ratio" in result.metrics
        assert result.dimension_name == "risk_adjusted"
        assert 0 <= result.score <= 100

    def test_risk_adjusted_with_benchmark(self) -> None:
        """Verify treynor/jensens_alpha/information_ratio computed when benchmark provided."""
        np.random.seed(103)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        benchmark = pd.Series(np.random.normal(0.0005, 0.015, 252), index=dates)

        evaluator = RiskAdjustedEvaluator()
        result = evaluator.evaluate(returns, benchmark_returns=benchmark)

        assert "treynor_ratio" in result.metrics
        assert "jensens_alpha" in result.metrics
        assert "information_ratio" in result.metrics

    def test_transaction_cost_evaluator(self) -> None:
        """Verify turnover and holding period computed from positions."""
        np.random.seed(104)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        # Low-turnover positions: mostly holding
        positions_arr = np.ones(252)
        positions_arr[::50] = 0  # Occasional exits
        positions = pd.DataFrame({"position": positions_arr}, index=dates)

        evaluator = TransactionCostEvaluator()
        result = evaluator.evaluate(returns, positions)

        assert "annual_turnover_rate" in result.metrics
        assert "avg_holding_period" in result.metrics
        assert "trade_frequency" in result.metrics
        assert result.dimension_name == "transaction_cost"

    def test_transaction_cost_high_turnover(self) -> None:
        """Verify low score for high-turnover positions."""
        np.random.seed(105)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        # Random positions that change every day
        positions_arr = np.random.choice([-1, 0, 1], size=252)
        positions = pd.DataFrame({"position": positions_arr}, index=dates)

        evaluator = TransactionCostEvaluator()
        result = evaluator.evaluate(returns, positions)

        assert result.metrics["annual_turnover_rate"] > 50.0
        assert result.score < 30.0

    def test_drawdown_risk_evaluator(self) -> None:
        """Verify max_drawdown, VaR, CVaR computed."""
        np.random.seed(106)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.0005, 0.02, 252), index=dates)

        evaluator = DrawdownRiskEvaluator()
        result = evaluator.evaluate(returns)

        assert "max_drawdown" in result.metrics
        assert "var_95" in result.metrics
        assert "cvar_95" in result.metrics
        assert "max_drawdown_duration" in result.metrics
        assert "pain_index" in result.metrics
        assert result.metrics["max_drawdown"] <= 0.0

    def test_drawdown_risk_no_drawdown(self) -> None:
        """All-positive returns should get high score."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        returns = pd.Series(np.full(100, 0.002), index=dates)

        evaluator = DrawdownRiskEvaluator()
        result = evaluator.evaluate(returns)

        assert result.metrics["max_drawdown"] == 0.0
        assert result.score > 80
        assert result.passed is True

    def test_efficiency_evaluator(self) -> None:
        """Verify profit_per_trade, expectancy computed."""
        np.random.seed(108)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        positions_arr = np.ones(252)
        positions_arr[::20] = 0
        positions = pd.DataFrame({"position": positions_arr}, index=dates)

        evaluator = EfficiencyEvaluator()
        result = evaluator.evaluate(returns, positions)

        assert "profit_per_trade" in result.metrics
        assert "expectancy" in result.metrics
        assert "edge_ratio" in result.metrics
        assert "payoff_ratio" in result.metrics
        assert result.dimension_name == "efficiency"

    def test_independence_evaluator_no_correlation(self) -> None:
        """Verify random returns get high independence score."""
        np.random.seed(109)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)

        evaluator = IndependenceEvaluator()
        result = evaluator.evaluate(returns)

        # No benchmark => correlation is 0 => high independence
        assert result.metrics["correlation_with_market"] == 0.0
        assert result.metrics["uniqueness_score"] == 1.0
        assert result.score >= 90

    def test_independence_evaluator_with_benchmark(self) -> None:
        """Verify correlation metrics when benchmark provided."""
        np.random.seed(110)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        market = pd.Series(np.random.normal(0.0005, 0.015, 252), index=dates)
        # Strategy that is correlated with market
        strategy_returns = market * 0.8 + pd.Series(
            np.random.normal(0, 0.005, 252), index=dates
        )

        evaluator = IndependenceEvaluator()
        result = evaluator.evaluate(strategy_returns, benchmark_returns=market)

        assert result.metrics["correlation_with_market"] > 0.5
        assert result.metrics["uniqueness_score"] < 0.5

    def test_coverage_evaluator(self) -> None:
        """Verify breadth and signal coverage computed."""
        np.random.seed(111)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        # Positions with some zeros
        positions_arr = np.ones(252)
        positions_arr[:50] = 0  # First 50 days no position
        positions = pd.DataFrame({"position": positions_arr}, index=dates)

        evaluator = CoverageEvaluator()
        result = evaluator.evaluate(returns, positions)

        assert "breadth" in result.metrics
        assert "signal_coverage_pct" in result.metrics
        assert result.metrics["breadth"] == 1.0
        expected_coverage = 202 / 252
        assert abs(result.metrics["signal_coverage_pct"] - expected_coverage) < 0.01

    def test_robustness_evaluator(self) -> None:
        """Verify rolling stability and bootstrap CI computed."""
        np.random.seed(112)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)

        evaluator = RobustnessEvaluator()
        result = evaluator.evaluate(returns)

        assert "rolling_sharpe_stability" in result.metrics
        assert "oos_is_ratio" in result.metrics
        assert "regime_stability" in result.metrics
        assert "bootstrap_mean_ci_lower" in result.metrics
        assert "bootstrap_mean_ci_upper" in result.metrics
        assert (
            result.metrics["bootstrap_mean_ci_lower"]
            <= result.metrics["bootstrap_mean_ci_upper"]
        )

    def test_robustness_stable_returns(self) -> None:
        """Stable consistent returns should get high robustness score."""
        np.random.seed(113)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        # Very low vol, consistent positive returns
        returns = pd.Series(
            np.random.normal(0.002, 0.002, 252), index=dates
        )

        evaluator = RobustnessEvaluator()
        result = evaluator.evaluate(returns)

        # With consistent positive returns, bootstrap CI should be above 0
        assert result.metrics["bootstrap_mean_ci_lower"] > 0
        assert result.passed is True


class TestAlphaValidator:
    """Tests for the AlphaValidator end-to-end flow."""

    def test_validator_default_config(self) -> None:
        """Instantiate with no args, verify default config."""
        validator = AlphaValidator()

        assert isinstance(validator.config, ValidationConfig)
        assert validator.config.min_sharpe == 0.5
        assert validator.config.min_overall_score == 50.0
        assert len(validator.config.dimension_weights) == 8

    def test_validator_validate_basic(self) -> None:
        """Run full validate() on a BacktestResult, verify 8 dimensions returned."""
        np.random.seed(200)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        bt = _make_backtest_result(returns)

        validator = AlphaValidator()
        report = validator.validate(bt)

        assert isinstance(report, ValidationReport)
        assert len(report.dimension_results) == 8
        dimension_names = [dr.dimension_name for dr in report.dimension_results]
        assert "return_ability" in dimension_names
        assert "risk_adjusted" in dimension_names
        assert "transaction_cost" in dimension_names
        assert "drawdown_risk" in dimension_names
        assert "efficiency" in dimension_names
        assert "independence" in dimension_names
        assert "coverage" in dimension_names
        assert "robustness" in dimension_names

    def test_validator_with_benchmark(self) -> None:
        """Pass benchmark_returns, verify risk-adjusted dimension uses it."""
        np.random.seed(201)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        benchmark = pd.Series(np.random.normal(0.0005, 0.015, 252), index=dates)
        bt = _make_backtest_result(returns)

        validator = AlphaValidator()
        report = validator.validate(bt, benchmark_returns=benchmark)

        risk_adjusted = next(
            dr for dr in report.dimension_results if dr.dimension_name == "risk_adjusted"
        )
        assert "treynor_ratio" in risk_adjusted.metrics
        assert "information_ratio" in risk_adjusted.metrics
        assert "jensens_alpha" in risk_adjusted.metrics

    def test_validator_with_existing_alphas(self) -> None:
        """Pass existing_alpha_returns, verify independence dimension uses it."""
        np.random.seed(202)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        existing = pd.DataFrame(
            {
                "alpha1": np.random.normal(0.0005, 0.015, 252),
                "alpha2": np.random.normal(0.0003, 0.018, 252),
            },
            index=dates,
        )
        bt = _make_backtest_result(returns)

        validator = AlphaValidator()
        report = validator.validate(bt, existing_alpha_returns=existing)

        independence = next(
            dr for dr in report.dimension_results if dr.dimension_name == "independence"
        )
        assert "marginal_contribution" in independence.metrics
        # With random data, marginal contribution should be high (low correlation)
        assert independence.metrics["marginal_contribution"] > 0.5

    def test_validator_high_quality_alpha(self) -> None:
        """Synthetic data designed to pass all dimensions."""
        np.random.seed(203)
        dates = pd.date_range("2020-01-01", periods=504, freq="B")
        # High Sharpe, low vol, consistent positive returns
        returns = pd.Series(np.random.normal(0.003, 0.005, 504), index=dates)
        # Low turnover positions
        positions_arr = np.ones(504)
        positions_arr[::100] = 0
        positions = pd.DataFrame({"position": positions_arr}, index=dates)
        bt = _make_backtest_result(returns, positions)

        validator = AlphaValidator()
        report = validator.validate(bt)

        assert report.overall_score > 60
        assert report.overall_passed is True

    def test_validator_low_quality_alpha(self) -> None:
        """Synthetic data designed to fail (negative returns, high drawdown)."""
        np.random.seed(204)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        # Negative mean returns, high vol
        returns = pd.Series(np.random.normal(-0.003, 0.04, 252), index=dates)
        # High turnover
        positions_arr = np.random.choice([-1, 0, 1], size=252)
        positions = pd.DataFrame({"position": positions_arr}, index=dates)
        bt = _make_backtest_result(returns, positions)

        validator = AlphaValidator()
        report = validator.validate(bt)

        assert report.overall_score < 50
        assert report.overall_passed is False


class TestValidationConfig:
    """Tests for ValidationConfig."""

    def test_default_config_values(self) -> None:
        """Verify default threshold values."""
        config = ValidationConfig()

        assert config.min_sharpe == 0.5
        assert config.min_sortino == 0.7
        assert config.min_annualized_return == 0.05
        assert config.max_drawdown_threshold == -0.30
        assert config.max_cvar_95 == 0.05
        assert config.max_annual_turnover == 20.0
        assert config.min_hit_rate == 0.45
        assert config.min_profit_factor == 1.0
        assert config.min_overall_score == 50.0
        assert len(config.dimension_weights) == 8

    def test_custom_config(self) -> None:
        """Create config with custom thresholds, verify validator uses them."""
        config = ValidationConfig(
            min_sharpe=1.0,
            min_overall_score=80.0,
        )
        validator = AlphaValidator(config=config)

        assert validator.config.min_sharpe == 1.0
        assert validator.config.min_overall_score == 80.0

    def test_config_affects_pass_fail(self) -> None:
        """Same data passes with lenient config but fails with strict config."""
        np.random.seed(300)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.015, 252), index=dates)
        bt = _make_backtest_result(returns)

        lenient_config = ValidationConfig(min_overall_score=20.0)
        strict_config = ValidationConfig(min_overall_score=95.0)

        lenient_validator = AlphaValidator(config=lenient_config)
        strict_validator = AlphaValidator(config=strict_config)

        lenient_report = lenient_validator.validate(bt)
        strict_report = strict_validator.validate(bt)

        # Same underlying scores
        assert lenient_report.overall_score == strict_report.overall_score
        # Different pass/fail
        assert lenient_report.overall_passed is True
        assert strict_report.overall_passed is False


class TestValidationReport:
    """Tests for ValidationReport."""

    def test_report_to_dict(self) -> None:
        """Verify to_dict() returns proper structure."""
        np.random.seed(400)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        bt = _make_backtest_result(returns)

        validator = AlphaValidator()
        report = validator.validate(bt)
        report_dict = report.to_dict()

        assert "overall_score" in report_dict
        assert "overall_passed" in report_dict
        assert "summary" in report_dict
        assert "timestamp" in report_dict
        assert "dimension_results" in report_dict
        assert isinstance(report_dict["dimension_results"], list)
        assert len(report_dict["dimension_results"]) == 8

        # Check structure of each dimension in dict
        for dr in report_dict["dimension_results"]:
            assert "dimension_name" in dr
            assert "metrics" in dr
            assert "score" in dr
            assert "passed" in dr
            assert "details" in dr

    def test_report_str(self) -> None:
        """Verify __str__ produces readable output."""
        np.random.seed(401)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        bt = _make_backtest_result(returns)

        validator = AlphaValidator()
        report = validator.validate(bt)
        report_str = str(report)

        assert "ALPHA VALIDATION REPORT" in report_str
        assert "Overall Score" in report_str
        assert "DIMENSION RESULTS" in report_str
        assert "return_ability" in report_str
        assert "risk_adjusted" in report_str

    def test_report_has_all_dimensions(self) -> None:
        """Verify all 8 dimension names present."""
        np.random.seed(402)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        bt = _make_backtest_result(returns)

        validator = AlphaValidator()
        report = validator.validate(bt)

        expected_dimensions = {
            "return_ability",
            "risk_adjusted",
            "transaction_cost",
            "drawdown_risk",
            "efficiency",
            "independence",
            "coverage",
            "robustness",
        }
        actual_dimensions = {dr.dimension_name for dr in report.dimension_results}
        assert actual_dimensions == expected_dimensions


class TestEdgeCases:
    """Tests for edge cases and degenerate inputs."""

    def test_constant_returns(self) -> None:
        """All zeros - should not crash."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        returns = pd.Series(np.zeros(100), index=dates)
        bt = _make_backtest_result(returns)

        validator = AlphaValidator()
        report = validator.validate(bt)

        assert isinstance(report, ValidationReport)
        assert len(report.dimension_results) == 8
        assert 0 <= report.overall_score <= 100

    def test_very_short_series(self) -> None:
        """5 data points - should not crash."""
        np.random.seed(501)
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 5), index=dates)
        bt = _make_backtest_result(returns)

        validator = AlphaValidator()
        report = validator.validate(bt)

        assert isinstance(report, ValidationReport)
        assert len(report.dimension_results) == 8

    def test_all_positive_returns(self) -> None:
        """No losing days."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        returns = pd.Series(np.full(100, 0.01), index=dates)
        bt = _make_backtest_result(returns)

        validator = AlphaValidator()
        report = validator.validate(bt)

        assert isinstance(report, ValidationReport)
        # All positive returns should produce good metrics
        return_ability = next(
            dr for dr in report.dimension_results if dr.dimension_name == "return_ability"
        )
        assert return_ability.metrics["hit_rate"] == 1.0

    def test_all_negative_returns(self) -> None:
        """No winning days."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        returns = pd.Series(np.full(100, -0.005), index=dates)
        bt = _make_backtest_result(returns)

        validator = AlphaValidator()
        report = validator.validate(bt)

        assert isinstance(report, ValidationReport)
        return_ability = next(
            dr for dr in report.dimension_results if dr.dimension_name == "return_ability"
        )
        assert return_ability.metrics["hit_rate"] == 0.0

    def test_no_trades(self) -> None:
        """Positions always 0."""
        np.random.seed(505)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        positions = pd.DataFrame({"position": np.zeros(100)}, index=dates)
        bt = _make_backtest_result(returns, positions)

        validator = AlphaValidator()
        report = validator.validate(bt)

        assert isinstance(report, ValidationReport)
        coverage = next(
            dr for dr in report.dimension_results if dr.dimension_name == "coverage"
        )
        assert coverage.metrics["signal_coverage_pct"] == 0.0
