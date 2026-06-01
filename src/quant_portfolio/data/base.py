"""Base data provider protocol for testability and dependency injection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataProvider(Protocol):
    """Protocol defining the interface for market data providers.

    Any data provider (live API, mock, CSV-based, etc.) should implement
    these methods to be usable with the portfolio construction framework.

    This protocol enables:
    - Mocking in unit tests without patching internals
    - Swapping data sources (Baostock, Tushare, CSV, database)
    - Dependency injection in the portfolio pipeline
    """

    def get_k_data(
        self,
        code: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjustflag: str = "3",
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch k-line (OHLCV) data for a given instrument.

        Parameters
        ----------
        code : str
            Instrument identifier (e.g., 'sh.600000').
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.
        frequency : str
            Data frequency: 'd' (daily), 'w' (weekly), 'm' (monthly).
        adjustflag : str
            Price adjustment flag.
        fields : str, optional
            Comma-separated field names.

        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex and requested market data columns.
        """
        ...

    def get_stock_list(self, date: Optional[str] = None) -> pd.DataFrame:
        """Fetch the list of available instruments.

        Parameters
        ----------
        date : str, optional
            Reference date for the stock list.

        Returns
        -------
        pd.DataFrame
            DataFrame with instrument codes and metadata.
        """
        ...


class BaseDataProvider(ABC):
    """Abstract base class for data providers.

    Use this when you want to enforce the interface via inheritance
    rather than structural typing (Protocol). Both approaches are valid;
    use Protocol for duck typing and this ABC for explicit contracts.
    """

    @abstractmethod
    def get_k_data(
        self,
        code: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjustflag: str = "3",
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch k-line (OHLCV) data for a given instrument."""
        ...

    @abstractmethod
    def get_stock_list(self, date: Optional[str] = None) -> pd.DataFrame:
        """Fetch the list of available instruments."""
        ...
