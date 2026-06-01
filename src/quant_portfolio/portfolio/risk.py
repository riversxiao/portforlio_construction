"""Risk analysis utilities for portfolio management."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Calculate Value at Risk (VaR).

    Parameters
    ----------
    returns : pd.Series
        Series of portfolio returns.
    confidence : float
        Confidence level (e.g., 0.95 for 95% VaR).
    method : str
        Calculation method: 'historical' or 'parametric'.

    Returns
    -------
    float
        VaR as a positive number representing potential loss.
    """
    if method == "historical":
        var = -np.percentile(returns.dropna(), (1 - confidence) * 100)
    elif method == "parametric":
        from scipy import stats

        z_score = stats.norm.ppf(1 - confidence)
        var = -(returns.mean() + z_score * returns.std())
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(var)


def calculate_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

    Parameters
    ----------
    returns : pd.Series
        Series of portfolio returns.
    confidence : float
        Confidence level.

    Returns
    -------
    float
        CVaR as a positive number.
    """
    var = calculate_var(returns, confidence, method="historical")
    # CVaR is the mean of losses beyond VaR
    tail_returns = returns[returns <= -var]
    if len(tail_returns) == 0:
        return var
    return float(-tail_returns.mean())


def calculate_volatility(
    returns: pd.Series,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> float:
    """Calculate portfolio volatility.

    Parameters
    ----------
    returns : pd.Series
        Series of portfolio returns.
    annualize : bool
        Whether to annualize the volatility.
    periods_per_year : int
        Number of periods per year for annualization.

    Returns
    -------
    float
        Portfolio volatility.
    """
    vol = returns.std()
    if annualize:
        vol *= np.sqrt(periods_per_year)
    return float(vol)


def calculate_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate portfolio beta relative to a benchmark.

    Parameters
    ----------
    returns : pd.Series
        Series of portfolio returns.
    benchmark_returns : pd.Series
        Series of benchmark returns.

    Returns
    -------
    float
        Portfolio beta.
    """
    # Align the series
    aligned = pd.concat(
        [returns, benchmark_returns], axis=1, join="inner"
    )
    aligned.columns = ["portfolio", "benchmark"]

    cov_matrix = aligned.cov()
    benchmark_var = cov_matrix.loc["benchmark", "benchmark"]

    if benchmark_var == 0:
        return 0.0

    beta = cov_matrix.loc["portfolio", "benchmark"] / benchmark_var
    return float(beta)


def calculate_tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> float:
    """Calculate tracking error relative to a benchmark.

    Parameters
    ----------
    returns : pd.Series
        Series of portfolio returns.
    benchmark_returns : pd.Series
        Series of benchmark returns.
    annualize : bool
        Whether to annualize.
    periods_per_year : int
        Number of periods per year.

    Returns
    -------
    float
        Tracking error.
    """
    active_returns = returns - benchmark_returns
    te = active_returns.std()
    if annualize:
        te *= np.sqrt(periods_per_year)
    return float(te)
