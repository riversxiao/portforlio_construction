"""Alpha Validation example.

Demonstrates how to validate a backtested strategy using the AlphaValidator,
which scores the strategy across 8 performance dimensions.

This example is self-contained and uses synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_portfolio.backtesting.engine import BacktestResult, Backtester
from quant_portfolio.strategies.registry import get_strategy
from quant_portfolio.validation import AlphaValidator, ValidationConfig


def generate_synthetic_data(n_periods: int = 504) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a trending component.

    Parameters
    ----------
    n_periods : int
        Number of trading days to generate.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with business day index.
    """
    np.random.seed(42)
    dates = pd.date_range("2019-01-01", periods=n_periods, freq="B")

    trend = np.linspace(0, 0.3, n_periods)
    noise = np.cumsum(np.random.normal(0, 0.015, n_periods))
    log_prices = trend + noise
    close = 100 * np.exp(log_prices)

    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
    open_ = close * (1 + np.random.normal(0, 0.005, n_periods))
    volume = np.random.randint(1_000_000, 10_000_000, n_periods).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def main() -> None:
    """Run alpha validation on a backtested momentum strategy."""
    print("=" * 60)
    print("ALPHA VALIDATION EXAMPLE")
    print("=" * 60)

    # --- Step 1: Run a backtest ---
    print("\n[Step 1] Running backtest with simple_momentum strategy...\n")

    data = generate_synthetic_data()
    print(f"  Data: {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}")

    StrategyClass = get_strategy("simple_momentum")
    strategy = StrategyClass()
    backtester = Backtester(
        strategy=strategy,
        initial_capital=1_000_000,
        commission=0.001,
        slippage=0.0005,
    )
    result = backtester.run(data)
    print(f"  Total return: {result.metrics['total_return']:.2%}")
    print(f"  Sharpe ratio: {result.metrics['sharpe_ratio']:.3f}")
    print(f"  Max drawdown: {result.metrics['max_drawdown']:.2%}")

    # --- Step 2: Validate with default config ---
    print("\n[Step 2] Validating with AlphaValidator (default config)...\n")

    validator = AlphaValidator()
    report = validator.validate(result)
    print(report)

    # --- Step 3: Validate with custom config (stricter thresholds) ---
    print("\n[Step 3] Validating with strict custom config...\n")

    strict_config = ValidationConfig(
        min_sharpe=1.5,
        max_drawdown_threshold=-0.15,
        max_annual_turnover=10.0,
        min_overall_score=70.0,
    )
    strict_validator = AlphaValidator(config=strict_config)
    strict_report = strict_validator.validate(result)

    print(f"  Strict validation overall score: {strict_report.overall_score:.1f}")
    print(f"  Strict validation passed: {strict_report.overall_passed}")

    # --- Step 4: Validate with benchmark returns ---
    print("\n[Step 4] Validating with benchmark returns...\n")

    np.random.seed(99)
    benchmark_returns = pd.Series(
        np.random.normal(0.0003, 0.012, len(result.returns)),
        index=result.returns.index,
    )

    report_with_benchmark = validator.validate(result, benchmark_returns=benchmark_returns)

    # Find the risk-adjusted dimension result
    risk_adjusted = next(
        dr for dr in report_with_benchmark.dimension_results
        if dr.dimension_name == "risk_adjusted"
    )
    print(f"  Information Ratio: {risk_adjusted.metrics.get('information_ratio', 'N/A'):.3f}")
    print(f"  Treynor Ratio: {risk_adjusted.metrics.get('treynor_ratio', 'N/A'):.3f}")
    print(f"  Jensen's Alpha: {risk_adjusted.metrics.get('jensens_alpha', 'N/A'):.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Default config pass: {report.overall_passed} (score={report.overall_score:.1f})")
    print(f"  Strict config pass:  {strict_report.overall_passed} (score={strict_report.overall_score:.1f})")
    print(f"  With benchmark pass: {report_with_benchmark.overall_passed} (score={report_with_benchmark.overall_score:.1f})")
    print("\nDone.")


if __name__ == "__main__":
    main()
