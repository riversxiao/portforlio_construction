"""Dimension evaluators for alpha validation."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from quant_portfolio.validation.report import DimensionResult


def _clip_score(score: float) -> float:
    """Clip score to [0, 100] range."""
    return float(np.clip(score, 0.0, 100.0))


class ReturnAbilityEvaluator:
    """Evaluates return generation ability of a strategy.

    Computes total return, annualized return, cumulative return, hit rate,
    profit factor, and average win/loss ratio.
    """

    def evaluate(
        self,
        returns: pd.Series,
        positions: Optional[pd.DataFrame] = None,
        periods_per_year: int = 252,
    ) -> DimensionResult:
        """Evaluate return ability dimension.

        Parameters
        ----------
        returns : pd.Series
            Strategy returns series.
        positions : pd.DataFrame, optional
            Position history.
        periods_per_year : int
            Trading periods per year.

        Returns
        -------
        DimensionResult
            Evaluation result with metrics and score.
        """
        total_return = float((1 + returns).prod() - 1)
        n_periods = len(returns)
        ann_return = float(
            (1 + total_return) ** (periods_per_year / max(n_periods, 1)) - 1
        )
        cumulative_return = total_return

        wins = returns[returns > 0]
        losses = returns[returns < 0]
        hit_rate = float(len(wins) / max(len(returns), 1))

        gross_profits = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_losses = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = (
            gross_profits / gross_losses if gross_losses > 0
            else (10.0 if gross_profits > 0 else 0.0)
        )

        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
        avg_win_loss_ratio = (
            avg_win / avg_loss if avg_loss > 0
            else (10.0 if avg_win > 0 else 0.0)
        )

        metrics: Dict[str, float] = {
            "total_return": total_return,
            "annualized_return": ann_return,
            "cumulative_return": cumulative_return,
            "hit_rate": hit_rate,
            "profit_factor": profit_factor,
            "avg_win_loss_ratio": avg_win_loss_ratio,
        }

        # Scoring: weighted combination
        # hit_rate contribution: 50%=0, 60%=100
        hit_score = _clip_score((hit_rate - 0.50) / 0.10 * 100.0)
        # profit_factor contribution: 1.0=0, 2.0=100
        pf_score = _clip_score((profit_factor - 1.0) / 1.0 * 100.0)
        # annualized_return contribution
        ret_score = _clip_score(ann_return / 0.20 * 100.0)

        score = _clip_score(0.4 * ret_score + 0.3 * hit_score + 0.3 * pf_score)
        passed = score >= 30.0

        details = (
            f"Ann. return: {ann_return:.2%}, "
            f"Hit rate: {hit_rate:.2%}, "
            f"Profit factor: {profit_factor:.2f}"
        )

        return DimensionResult(
            dimension_name="return_ability",
            metrics=metrics,
            score=score,
            passed=passed,
            details=details,
        )


class RiskAdjustedEvaluator:
    """Evaluates risk-adjusted performance metrics."""

    def evaluate(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> DimensionResult:
        """Evaluate risk-adjusted performance.

        Parameters
        ----------
        returns : pd.Series
            Strategy returns series.
        benchmark_returns : pd.Series, optional
            Benchmark returns for relative metrics.
        risk_free_rate : float
            Annualized risk-free rate.
        periods_per_year : int
            Trading periods per year.

        Returns
        -------
        DimensionResult
            Evaluation result with metrics and score.
        """
        excess_returns = returns - risk_free_rate / periods_per_year
        vol = returns.std()

        # Sharpe ratio
        sharpe = (
            float(excess_returns.mean() / vol * np.sqrt(periods_per_year))
            if vol > 0
            else 0.0
        )

        # Sortino ratio
        downside = excess_returns[excess_returns < 0]
        downside_std = downside.std() if len(downside) > 0 else 0.0
        sortino = (
            float(excess_returns.mean() / downside_std * np.sqrt(periods_per_year))
            if downside_std > 0
            else 0.0
        )

        # Omega ratio (threshold=0)
        gains = returns[returns > 0].sum()
        loss = abs(returns[returns <= 0].sum())
        omega = float(gains / loss) if loss > 0 else 0.0

        metrics: Dict[str, float] = {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "omega_ratio": omega,
        }

        # Information ratio, Treynor ratio, Jensen's alpha if benchmark
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            aligned = pd.concat(
                [returns, benchmark_returns], axis=1, join="inner"
            )
            aligned.columns = ["strategy", "benchmark"]
            active = aligned["strategy"] - aligned["benchmark"]
            te = active.std()
            info_ratio = (
                float(active.mean() / te * np.sqrt(periods_per_year))
                if te > 0
                else 0.0
            )
            metrics["information_ratio"] = info_ratio

            # Beta and Treynor
            cov_matrix = aligned.cov()
            bench_var = cov_matrix.loc["benchmark", "benchmark"]
            beta = (
                float(cov_matrix.loc["strategy", "benchmark"] / bench_var)
                if bench_var > 0
                else 0.0
            )
            treynor = (
                float(
                    (excess_returns.mean() * periods_per_year) / beta
                )
                if beta != 0
                else 0.0
            )
            metrics["treynor_ratio"] = treynor

            # Jensen's alpha
            bench_excess = (
                aligned["benchmark"] - risk_free_rate / periods_per_year
            )
            jensens_alpha = float(
                (excess_returns.mean() - beta * bench_excess.mean())
                * periods_per_year
            )
            metrics["jensens_alpha"] = jensens_alpha

        # Scoring: Sharpe 0=0, 1.0=50, 2.0=100; Sortino contribution
        sharpe_score = _clip_score(sharpe / 2.0 * 100.0)
        sortino_score = _clip_score(sortino / 3.0 * 100.0)
        score = _clip_score(0.6 * sharpe_score + 0.4 * sortino_score)
        passed = score >= 25.0

        details = (
            f"Sharpe: {sharpe:.2f}, "
            f"Sortino: {sortino:.2f}, "
            f"Omega: {omega:.2f}"
        )

        return DimensionResult(
            dimension_name="risk_adjusted",
            metrics=metrics,
            score=score,
            passed=passed,
            details=details,
        )


class TransactionCostEvaluator:
    """Evaluates transaction cost and turnover characteristics.

    Notes
    -----
    The metrics ``estimated_implementation_shortfall``, ``capacity_score``,
    and ``market_impact_score`` are simplified estimates derived from the
    annual turnover rate alone. They serve as placeholders for more
    sophisticated volume-aware calculations that require order-book or
    traded-volume data not available in a standard backtest result.
    """

    def evaluate(
        self,
        returns: pd.Series,
        positions: pd.DataFrame,
        periods_per_year: int = 252,
    ) -> DimensionResult:
        """Evaluate transaction cost dimension.

        Parameters
        ----------
        returns : pd.Series
            Strategy returns series.
        positions : pd.DataFrame
            Position history with a 'position' column.
        periods_per_year : int
            Trading periods per year.

        Returns
        -------
        DimensionResult
            Evaluation result with metrics and score.
        """
        if "position" in positions.columns:
            pos_series = positions["position"]
        else:
            pos_series = positions.iloc[:, 0]

        pos_changes = pos_series.diff().abs()
        n_periods = len(positions)
        n_years = max(n_periods / periods_per_year, 1e-6)

        # Annual turnover rate
        total_turnover = float(pos_changes.sum())
        annual_turnover = total_turnover / n_years

        # Trade frequency
        trades = pos_changes[pos_changes > 0]
        trade_frequency = float(len(trades) / max(n_periods, 1))

        # Average holding period
        avg_holding_period = (
            1.0 / trade_frequency if trade_frequency > 0 else float(n_periods)
        )

        # Estimated implementation shortfall (simplified: turnover-based proxy,
        # placeholder for volume-aware calculation)
        estimated_shortfall = annual_turnover * 0.001

        # Capacity score: lower turnover = higher capacity
        # (simplified: turnover-based proxy, placeholder for volume-aware calculation)
        capacity_score = _clip_score(100.0 - annual_turnover * 2.0)

        # Market impact score
        # (simplified: turnover-based proxy, placeholder for volume-aware calculation)
        market_impact_score = _clip_score(100.0 - annual_turnover * 1.5)

        metrics: Dict[str, float] = {
            "annual_turnover_rate": annual_turnover,
            "avg_holding_period": avg_holding_period,
            "trade_frequency": trade_frequency,
            "estimated_implementation_shortfall": estimated_shortfall,
            "capacity_score": capacity_score,
            "market_impact_score": market_impact_score,
        }

        # Scoring: turnover<5=100, turnover>50=0 linearly
        score = _clip_score((50.0 - annual_turnover) / 45.0 * 100.0)
        passed = annual_turnover <= 50.0

        details = (
            f"Annual turnover: {annual_turnover:.1f}, "
            f"Avg holding: {avg_holding_period:.1f} periods, "
            f"Trade freq: {trade_frequency:.3f}"
        )

        return DimensionResult(
            dimension_name="transaction_cost",
            metrics=metrics,
            score=score,
            passed=passed,
            details=details,
        )


class DrawdownRiskEvaluator:
    """Evaluates drawdown and tail risk characteristics."""

    def evaluate(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> DimensionResult:
        """Evaluate drawdown risk dimension.

        Parameters
        ----------
        returns : pd.Series
            Strategy returns series.
        periods_per_year : int
            Trading periods per year.

        Returns
        -------
        DimensionResult
            Evaluation result with metrics and score.
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        mdd = float(drawdown.min())

        # Max drawdown duration
        is_underwater = drawdown < 0
        if is_underwater.any():
            groups = (~is_underwater).cumsum()
            underwater_groups = groups[is_underwater]
            if len(underwater_groups) > 0:
                max_dd_duration = int(
                    underwater_groups.value_counts().max()
                )
            else:
                max_dd_duration = 0
        else:
            max_dd_duration = 0

        # VaR 95
        var_95 = float(-np.percentile(returns.dropna(), 5))

        # CVaR 95
        threshold = np.percentile(returns.dropna(), 5)
        tail = returns[returns <= threshold]
        cvar_95 = float(-tail.mean()) if len(tail) > 0 else var_95

        # Drawdown recovery ratio
        recovery_periods = len(returns) - max_dd_duration
        drawdown_recovery_ratio = (
            float(recovery_periods / max(max_dd_duration, 1))
        )

        # Underwater time percentage
        underwater_time_pct = float(is_underwater.sum() / max(len(returns), 1))

        # Pain index (mean of absolute drawdowns)
        pain_index = float(drawdown.abs().mean())

        metrics: Dict[str, float] = {
            "max_drawdown": mdd,
            "max_drawdown_duration": float(max_dd_duration),
            "var_95": var_95,
            "cvar_95": cvar_95,
            "drawdown_recovery_ratio": drawdown_recovery_ratio,
            "underwater_time_pct": underwater_time_pct,
            "pain_index": pain_index,
        }

        # Scoring: max_drawdown -0.05=100, -0.50=0; CVaR contribution
        dd_score = _clip_score((abs(mdd) - 0.50) / (0.05 - 0.50) * 100.0)
        cvar_score = _clip_score((0.10 - cvar_95) / 0.10 * 100.0)
        score = _clip_score(0.6 * dd_score + 0.4 * cvar_score)
        passed = mdd > -0.50

        details = (
            f"Max DD: {mdd:.2%}, "
            f"DD duration: {max_dd_duration} periods, "
            f"CVaR95: {cvar_95:.4f}"
        )

        return DimensionResult(
            dimension_name="drawdown_risk",
            metrics=metrics,
            score=score,
            passed=passed,
            details=details,
        )


class EfficiencyEvaluator:
    """Evaluates trading efficiency metrics."""

    def evaluate(
        self,
        returns: pd.Series,
        positions: pd.DataFrame,
        periods_per_year: int = 252,
    ) -> DimensionResult:
        """Evaluate efficiency dimension.

        Parameters
        ----------
        returns : pd.Series
            Strategy returns series.
        positions : pd.DataFrame
            Position history.
        periods_per_year : int
            Trading periods per year.

        Returns
        -------
        DimensionResult
            Evaluation result with metrics and score.
        """
        if "position" in positions.columns:
            pos_series = positions["position"]
        else:
            pos_series = positions.iloc[:, 0]

        pos_changes = pos_series.diff().abs()
        trades_mask = pos_changes > 0
        n_trades = int(trades_mask.sum())

        # Profit per trade
        total_return = float((1 + returns).prod() - 1)
        profit_per_trade = total_return / max(n_trades, 1)

        # Return per unit turnover
        total_turnover = float(pos_changes.sum())
        return_per_unit_turnover = (
            total_return / total_turnover if total_turnover > 0 else 0.0
        )

        # Edge ratio (average win / average loss)
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
        edge_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        # Expectancy
        hit_rate = len(wins) / max(len(returns), 1)
        expectancy = hit_rate * avg_win - (1 - hit_rate) * avg_loss

        # Payoff ratio
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        metrics: Dict[str, float] = {
            "profit_per_trade": profit_per_trade,
            "return_per_unit_turnover": return_per_unit_turnover,
            "edge_ratio": edge_ratio,
            "expectancy": expectancy,
            "payoff_ratio": payoff_ratio,
        }

        # Scoring: positive expectancy and edge_ratio > 1 contribute
        exp_score = _clip_score(expectancy / 0.005 * 50.0 + 50.0)
        edge_score = _clip_score((edge_ratio - 0.5) / 1.5 * 100.0)
        score = _clip_score(0.5 * exp_score + 0.5 * edge_score)
        passed = expectancy > 0

        details = (
            f"Expectancy: {expectancy:.5f}, "
            f"Edge ratio: {edge_ratio:.2f}, "
            f"Profit/trade: {profit_per_trade:.5f}"
        )

        return DimensionResult(
            dimension_name="efficiency",
            metrics=metrics,
            score=score,
            passed=passed,
            details=details,
        )


class IndependenceEvaluator:
    """Evaluates independence from market and existing alphas."""

    def evaluate(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        existing_alpha_returns: Optional[pd.DataFrame] = None,
    ) -> DimensionResult:
        """Evaluate independence dimension.

        Parameters
        ----------
        returns : pd.Series
            Strategy returns series.
        benchmark_returns : pd.Series, optional
            Market/benchmark returns.
        existing_alpha_returns : pd.DataFrame, optional
            Returns of existing alpha strategies (columns are alphas).

        Returns
        -------
        DimensionResult
            Evaluation result with metrics and score.
        """
        metrics: Dict[str, float] = {}

        # Correlation with market
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            aligned = pd.concat(
                [returns, benchmark_returns], axis=1, join="inner"
            )
            aligned.columns = ["strategy", "benchmark"]
            corr_market = float(aligned["strategy"].corr(aligned["benchmark"]))
        else:
            corr_market = 0.0
        metrics["correlation_with_market"] = corr_market

        # Factor correlations (simplified proxies)
        factor_corrs: Dict[str, float] = {}
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            factor_corrs["market"] = corr_market
        metrics["abs_market_correlation"] = float(abs(corr_market))

        # Marginal contribution
        if (
            existing_alpha_returns is not None
            and len(existing_alpha_returns) > 0
        ):
            aligned = pd.concat(
                [returns, existing_alpha_returns], axis=1, join="inner"
            )
            existing_cols = existing_alpha_returns.columns.tolist()
            corrs = [
                abs(float(returns.corr(existing_alpha_returns[c])))
                for c in existing_cols
                if c in aligned.columns
            ]
            avg_corr = float(np.mean(corrs)) if corrs else 0.0
            marginal = 1.0 - avg_corr
        else:
            marginal = 1.0

        metrics["marginal_contribution"] = marginal

        # Uniqueness score
        uniqueness = 1.0 - abs(corr_market)
        metrics["uniqueness_score"] = uniqueness

        # Scoring: low absolute correlation = high score
        corr_score = _clip_score((1.0 - abs(corr_market)) * 100.0)
        marginal_score = _clip_score(marginal * 100.0)
        score = _clip_score(0.5 * corr_score + 0.5 * marginal_score)
        passed = abs(corr_market) < 0.8

        details = (
            f"Market corr: {corr_market:.3f}, "
            f"Uniqueness: {uniqueness:.3f}, "
            f"Marginal contribution: {marginal:.3f}"
        )

        return DimensionResult(
            dimension_name="independence",
            metrics=metrics,
            score=score,
            passed=passed,
            details=details,
        )


class CoverageEvaluator:
    """Evaluates signal coverage and breadth."""

    def evaluate(
        self,
        returns: pd.Series,
        positions: pd.DataFrame,
    ) -> DimensionResult:
        """Evaluate coverage dimension.

        Parameters
        ----------
        returns : pd.Series
            Strategy returns series.
        positions : pd.DataFrame
            Position history.

        Returns
        -------
        DimensionResult
            Evaluation result with metrics and score.
        """
        if "position" in positions.columns:
            pos_series = positions["position"]
            # Single-asset case
            breadth = 1.0
            signal_coverage = float(
                (pos_series != 0).sum() / max(len(pos_series), 1)
            )
        else:
            # Multi-asset: count unique non-zero position columns
            non_zero_cols = (positions != 0).any()
            breadth = float(non_zero_cols.sum())
            signal_coverage = float(
                (positions != 0).any(axis=1).sum() / max(len(positions), 1)
            )

        active_ratio = signal_coverage

        metrics: Dict[str, float] = {
            "breadth": breadth,
            "signal_coverage_pct": signal_coverage,
            "active_ratio": active_ratio,
        }

        # Scoring: for single-asset, score based on signal_coverage_pct
        score = _clip_score(signal_coverage * 100.0)
        passed = signal_coverage >= 0.1

        details = (
            f"Breadth: {breadth:.0f}, "
            f"Signal coverage: {signal_coverage:.2%}, "
            f"Active ratio: {active_ratio:.2%}"
        )

        return DimensionResult(
            dimension_name="coverage",
            metrics=metrics,
            score=score,
            passed=passed,
            details=details,
        )


class RobustnessEvaluator:
    """Evaluates strategy robustness and stability."""

    def evaluate(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> DimensionResult:
        """Evaluate robustness dimension.

        Parameters
        ----------
        returns : pd.Series
            Strategy returns series.
        periods_per_year : int
            Trading periods per year.

        Returns
        -------
        DimensionResult
            Evaluation result with metrics and score.
        """
        n = len(returns)

        # Rolling Sharpe stability
        window = min(periods_per_year // 4, max(n // 4, 10))
        if n > window:
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan)
            rolling_sharpe = rolling_sharpe.dropna()
            if len(rolling_sharpe) > 0 and rolling_sharpe.std() > 0:
                sharpe_stability = float(
                    1.0
                    - rolling_sharpe.std()
                    / (abs(rolling_sharpe.mean()) + 1e-8)
                )
                sharpe_stability = max(0.0, min(1.0, sharpe_stability))
            else:
                sharpe_stability = 0.5
        else:
            sharpe_stability = 0.5

        # OOS/IS ratio: second-half Sharpe / first-half Sharpe
        mid = n // 2
        if mid > 10:
            first_half = returns.iloc[:mid]
            second_half = returns.iloc[mid:]
            std_1 = first_half.std()
            std_2 = second_half.std()
            sharpe_1 = (
                float(first_half.mean() / std_1 * np.sqrt(periods_per_year))
                if std_1 > 0
                else 0.0
            )
            sharpe_2 = (
                float(second_half.mean() / std_2 * np.sqrt(periods_per_year))
                if std_2 > 0
                else 0.0
            )
            # Handle sign differences explicitly
            if abs(sharpe_1) <= 1e-8:
                oos_is_ratio = 0.0
            elif sharpe_1 < 0 and sharpe_2 >= 0:
                # Improved from negative to positive: not directly comparable
                oos_is_ratio = 0.0
            elif sharpe_1 < 0 and sharpe_2 < 0:
                # Both negative: compare absolute magnitudes (lower abs = better OOS)
                oos_is_ratio = abs(sharpe_1) / abs(sharpe_2)
            else:
                oos_is_ratio = sharpe_2 / sharpe_1
        else:
            oos_is_ratio = 1.0

        # Regime stability (simplified: compare positive vs negative market)
        median_ret = returns.median()
        bull = returns[returns >= median_ret]
        bear = returns[returns < median_ret]
        bull_sharpe = (
            float(bull.mean() / bull.std()) if bull.std() > 0 else 0.0
        )
        bear_sharpe = (
            float(bear.mean() / bear.std()) if bear.std() > 0 else 0.0
        )
        regime_stability = (
            1.0
            - abs(bull_sharpe - bear_sharpe)
            / (abs(bull_sharpe) + abs(bear_sharpe) + 1e-8)
        )
        regime_stability = max(0.0, min(1.0, regime_stability))

        # Bootstrap confidence intervals
        n_bootstrap = 1000
        rng = np.random.default_rng(42)
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(returns.values, size=n, replace=True)
            bootstrap_means.append(float(np.mean(sample)))
        bootstrap_means_arr = np.array(bootstrap_means)
        ci_lower = float(np.percentile(bootstrap_means_arr, 2.5))
        ci_upper = float(np.percentile(bootstrap_means_arr, 97.5))

        metrics: Dict[str, float] = {
            "rolling_sharpe_stability": sharpe_stability,
            "oos_is_ratio": oos_is_ratio,
            "regime_stability": regime_stability,
            "bootstrap_mean_ci_lower": ci_lower,
            "bootstrap_mean_ci_upper": ci_upper,
        }

        # Scoring
        stability_score = _clip_score(sharpe_stability * 100.0)
        oos_score = _clip_score(
            (1.0 - abs(1.0 - oos_is_ratio)) * 100.0
        )
        # CI not including 0 is good
        ci_score = 100.0 if ci_lower > 0 else _clip_score(
            (ci_lower / (ci_lower - ci_upper + 1e-10) + 0.5) * 100.0
        )
        score = _clip_score(
            0.4 * stability_score + 0.3 * oos_score + 0.3 * ci_score
        )
        passed = sharpe_stability > 0.3

        details = (
            f"Sharpe stability: {sharpe_stability:.3f}, "
            f"OOS/IS ratio: {oos_is_ratio:.3f}, "
            f"Bootstrap CI: [{ci_lower:.5f}, {ci_upper:.5f}]"
        )

        return DimensionResult(
            dimension_name="robustness",
            metrics=metrics,
            score=score,
            passed=passed,
            details=details,
        )
