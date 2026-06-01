"""Alpha validator that orchestrates all dimension evaluations."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from quant_portfolio.backtesting.engine import BacktestResult
from quant_portfolio.validation.config import ValidationConfig
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
from quant_portfolio.validation.report import DimensionResult, ValidationReport


class AlphaValidator:
    """Validates alpha strategies across 8 performance dimensions.

    Parameters
    ----------
    config : ValidationConfig, optional
        Configuration with thresholds and dimension weights.
        Uses default config if None.
    """

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        self.config = config or ValidationConfig()
        self._return_evaluator = ReturnAbilityEvaluator()
        self._risk_adjusted_evaluator = RiskAdjustedEvaluator()
        self._transaction_cost_evaluator = TransactionCostEvaluator()
        self._drawdown_evaluator = DrawdownRiskEvaluator()
        self._efficiency_evaluator = EfficiencyEvaluator()
        self._independence_evaluator = IndependenceEvaluator()
        self._coverage_evaluator = CoverageEvaluator()
        self._robustness_evaluator = RobustnessEvaluator()

    def validate(
        self,
        backtest_result: BacktestResult,
        benchmark_returns: Optional[pd.Series] = None,
        existing_alpha_returns: Optional[pd.DataFrame] = None,
        periods_per_year: int = 252,
    ) -> ValidationReport:
        """Run full validation across all 8 dimensions.

        Parameters
        ----------
        backtest_result : BacktestResult
            Result from a strategy backtest.
        benchmark_returns : pd.Series, optional
            Benchmark returns for relative metrics.
        existing_alpha_returns : pd.DataFrame, optional
            Returns of existing alpha strategies for independence check.
        periods_per_year : int
            Number of trading periods per year.

        Returns
        -------
        ValidationReport
            Complete validation report with scores and metrics.
        """
        returns = backtest_result.returns
        positions = backtest_result.positions

        dimension_results = []

        # 1. Return ability
        result = self._return_evaluator.evaluate(
            returns, positions, periods_per_year
        )
        dimension_results.append(result)

        # 2. Risk-adjusted
        result = self._risk_adjusted_evaluator.evaluate(
            returns, benchmark_returns, periods_per_year=periods_per_year
        )
        dimension_results.append(result)

        # 3. Transaction cost
        result = self._transaction_cost_evaluator.evaluate(
            returns, positions, periods_per_year
        )
        dimension_results.append(result)

        # 4. Drawdown risk
        result = self._drawdown_evaluator.evaluate(returns, periods_per_year)
        dimension_results.append(result)

        # 5. Efficiency
        result = self._efficiency_evaluator.evaluate(
            returns, positions, periods_per_year
        )
        dimension_results.append(result)

        # 6. Independence
        result = self._independence_evaluator.evaluate(
            returns, benchmark_returns, existing_alpha_returns
        )
        dimension_results.append(result)

        # 7. Coverage
        result = self._coverage_evaluator.evaluate(returns, positions)
        dimension_results.append(result)

        # 8. Robustness
        result = self._robustness_evaluator.evaluate(returns, periods_per_year)
        dimension_results.append(result)

        # Compute overall score as weighted average
        weights = self.config.dimension_weights
        total_weight = 0.0
        weighted_score = 0.0
        for dr in dimension_results:
            w = weights.get(dr.dimension_name, 1.0)
            weighted_score += dr.score * w
            total_weight += w

        overall_score = (
            weighted_score / total_weight if total_weight > 0 else 0.0
        )
        overall_passed = overall_score >= self.config.min_overall_score

        # Build summary
        passed_dims = sum(1 for dr in dimension_results if dr.passed)
        total_dims = len(dimension_results)
        summary = (
            f"Score: {overall_score:.1f}/100. "
            f"Passed {passed_dims}/{total_dims} dimensions. "
            f"{'PASSED' if overall_passed else 'FAILED'} overall validation."
        )

        return ValidationReport(
            overall_score=overall_score,
            overall_passed=overall_passed,
            dimension_results=dimension_results,
            summary=summary,
            timestamp=datetime.now(),
        )
