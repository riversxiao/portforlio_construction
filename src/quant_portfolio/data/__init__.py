"""Data layer for fetching and processing market data."""

from quant_portfolio.data.base import BaseDataProvider, DataProvider
from quant_portfolio.data.baostock_provider import BaostockProvider
from quant_portfolio.data.data_utils import (
    calculate_returns,
    clean_data,
    resample_data,
)

__all__ = [
    "BaseDataProvider",
    "BaostockProvider",
    "DataProvider",
    "calculate_returns",
    "clean_data",
    "resample_data",
]
