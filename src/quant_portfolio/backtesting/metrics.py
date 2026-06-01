"""Standalone performance metrics functions for backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized return.

    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns.
    periods_per_year : int
        Number of periods per year (252 for daily, 52 for weekly, 12 for monthly).

    Returns
    -------
    float
        Annualized return.
    """
    total_return = (1 + returns).prod()
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0
    return float(total_return ** (periods_per_year / n_periods) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility.

    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns.
    periods_per_year : int
        Number of periods per year.

    Returns
    -------
    float
        Annualized volatility (standard deviation).
    """
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns.
    risk_free_rate : float
        Annualized risk-free rate.
    periods_per_year : int
        Number of periods per year.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() == 0:
        return 0.0
    return float(
        excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    )


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sortino ratio.

    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns.
    risk_free_rate : float
        Annualized risk-free rate.
    periods_per_year : int
        Number of periods per year.

    Returns
    -------
    float
        Annualized Sortino ratio.
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    downside_std = downside_returns.std()
    return float(
        excess_returns.mean() / downside_std * np.sqrt(periods_per_year)
    )


def max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown.

    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns.

    Returns
    -------
    float
        Maximum drawdown as a negative number (e.g., -0.20 for 20% drawdown).
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown).

    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns.
    periods_per_year : int
        Number of periods per year.

    Returns
    -------
    float
        Calmar ratio.
    """
    ann_ret = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)

    if mdd == 0:
        return 0.0

    return float(ann_ret / abs(mdd))


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate information ratio.

    Parameters
    ----------
    returns : pd.Series
        Series of portfolio returns.
    benchmark_returns : pd.Series
        Series of benchmark returns.
    periods_per_year : int
        Number of periods per year.

    Returns
    -------
    float
        Annualized information ratio.
    """
    active_returns = returns - benchmark_returns
    tracking_err = active_returns.std()

    if tracking_err == 0:
        return 0.0

    return float(
        active_returns.mean() / tracking_err * np.sqrt(periods_per_year)
    )
