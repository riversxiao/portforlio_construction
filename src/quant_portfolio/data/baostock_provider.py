"""Baostock API wrapper for fetching market data."""

from __future__ import annotations

from typing import Optional

import pandas as pd


class BaostockProvider:
    """Wrapper around the Baostock API for fetching A-share market data.

    Provides methods for fetching daily/weekly/monthly k-line data,
    stock lists, industry classification, and fundamental data.
    """

    def __init__(self) -> None:
        self._logged_in = False

    def login(self) -> None:
        """Log in to the Baostock server."""
        import baostock as bs

        result = bs.login()
        if result.error_code != "0":
            raise ConnectionError(
                f"Baostock login failed: {result.error_msg}"
            )
        self._logged_in = True

    def logout(self) -> None:
        """Log out from the Baostock server."""
        import baostock as bs

        bs.logout()
        self._logged_in = False

    def _ensure_login(self) -> None:
        """Ensure that a Baostock session is active."""
        if not self._logged_in:
            self.login()

    def get_k_data(
        self,
        code: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjustflag: str = "3",
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch k-line data for a given stock code.

        Parameters
        ----------
        code : str
            Stock code, e.g. 'sh.600000' or 'sz.000001'.
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.
        frequency : str
            Data frequency: 'd' (daily), 'w' (weekly), 'm' (monthly).
        adjustflag : str
            Adjustment flag: '1' (backward), '2' (forward), '3' (no adjust).
        fields : str, optional
            Comma-separated field names. Defaults to OHLCV fields.

        Returns
        -------
        pd.DataFrame
            DataFrame with requested k-line data.
        """
        import baostock as bs

        self._ensure_login()

        if fields is None:
            fields = "date,open,high,low,close,volume,amount"

        rs = bs.query_history_k_data_plus(
            code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag=adjustflag,
        )

        if rs.error_code != "0":
            raise RuntimeError(
                f"Failed to fetch k-data for {code}: {rs.error_msg}"
            )

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        df = pd.DataFrame(rows, columns=rs.fields)

        # Convert numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        return df

    def get_stock_list(self, date: Optional[str] = None) -> pd.DataFrame:
        """Fetch the list of all stocks on a given trade date.

        Parameters
        ----------
        date : str, optional
            Trade date in 'YYYY-MM-DD' format. Defaults to latest trade date.

        Returns
        -------
        pd.DataFrame
            DataFrame with stock codes and names.
        """
        import baostock as bs

        self._ensure_login()

        if date is None:
            date = pd.Timestamp.now().strftime("%Y-%m-%d")

        rs = bs.query_all_stock(day=date)
        if rs.error_code != "0":
            raise RuntimeError(
                f"Failed to fetch stock list: {rs.error_msg}"
            )

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        return pd.DataFrame(rows, columns=rs.fields)

    def get_industry_classification(self, code: str) -> pd.DataFrame:
        """Fetch industry classification for a stock.

        Parameters
        ----------
        code : str
            Stock code, e.g. 'sh.600000'.

        Returns
        -------
        pd.DataFrame
            DataFrame with industry classification information.
        """
        import baostock as bs

        self._ensure_login()

        rs = bs.query_stock_industry(code=code)
        if rs.error_code != "0":
            raise RuntimeError(
                f"Failed to fetch industry for {code}: {rs.error_msg}"
            )

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        return pd.DataFrame(rows, columns=rs.fields)

    def __enter__(self) -> "BaostockProvider":
        """Context manager entry."""
        self.login()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.logout()
