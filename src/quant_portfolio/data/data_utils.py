"""Utility functions for data cleaning, resampling, and returns calculation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def calculate_returns(
    prices: pd.DataFrame,
    method: str = "simple",
    periods: int = 1,
) -> pd.DataFrame:
    """Calculate returns from price data.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with DatetimeIndex.
    method : str
        Return calculation method: 'simple' or 'log'.
    periods : int
        Number of periods to compute returns over.

    Returns
    -------
    pd.DataFrame
        DataFrame of returns.
    """
    if method == "log":
        returns = np.log(prices / prices.shift(periods))
    elif method == "simple":
        returns = prices.pct_change(periods=periods)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'simple' or 'log'.")

    return returns.dropna()


def clean_data(
    df: pd.DataFrame,
    method: str = "ffill",
    drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """Clean market data by handling missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with potential missing values.
    method : str
        Fill method: 'ffill', 'bfill', 'interpolate', or 'drop'.
    drop_threshold : float
        Drop columns with more than this fraction of missing values.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    # Drop columns with too many missing values
    missing_frac = df.isnull().mean()
    cols_to_keep = missing_frac[missing_frac <= drop_threshold].index
    df = df[cols_to_keep].copy()

    if method == "ffill":
        df = df.ffill().bfill()
    elif method == "bfill":
        df = df.bfill().ffill()
    elif method == "interpolate":
        df = df.interpolate(method="time").bfill().ffill()
    elif method == "drop":
        df = df.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")

    return df


def resample_data(
    df: pd.DataFrame,
    freq: str = "W",
    agg_method: Optional[str] = None,
) -> pd.DataFrame:
    """Resample time-series data to a different frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with DatetimeIndex.
    freq : str
        Target frequency: 'W' (weekly), 'M' (monthly), 'Q' (quarterly).
    agg_method : str, optional
        Aggregation method: 'last', 'mean', 'sum'. Defaults to 'last'.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame.
    """
    if agg_method is None:
        agg_method = "last"

    resampler = df.resample(freq)

    if agg_method == "last":
        return resampler.last()
    elif agg_method == "mean":
        return resampler.mean()
    elif agg_method == "sum":
        return resampler.sum()
    else:
        raise ValueError(f"Unknown agg_method: {agg_method}")
