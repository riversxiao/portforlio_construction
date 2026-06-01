"""Tests for data utilities and provider interfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.data.baostock_provider import BaostockProvider
from quant_portfolio.data.data_utils import (
    calculate_returns,
    clean_data,
    resample_data,
)


class TestCalculateReturns:
    """Tests for calculate_returns function."""

    def test_simple_returns(self, sample_prices: pd.DataFrame) -> None:
        """Test simple return calculation."""
        returns = calculate_returns(sample_prices, method="simple")
        assert len(returns) == len(sample_prices) - 1
        assert not returns.isnull().any().any()

    def test_log_returns(self, sample_prices: pd.DataFrame) -> None:
        """Test log return calculation."""
        returns = calculate_returns(sample_prices, method="log")
        assert len(returns) == len(sample_prices) - 1
        assert not returns.isnull().any().any()

    def test_invalid_method(self, sample_prices: pd.DataFrame) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            calculate_returns(sample_prices, method="invalid")

    def test_multi_period_returns(self, sample_prices: pd.DataFrame) -> None:
        """Test multi-period return calculation."""
        returns = calculate_returns(sample_prices, periods=5)
        assert len(returns) == len(sample_prices) - 5


class TestCleanData:
    """Tests for clean_data function."""

    def test_ffill_method(self) -> None:
        """Test forward fill cleaning."""
        df = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]},
            index=pd.date_range("2020-01-01", periods=3),
        )
        cleaned = clean_data(df, method="ffill")
        assert not cleaned.isnull().any().any()

    def test_drop_threshold(self) -> None:
        """Test that columns with too many NaN values are dropped."""
        df = pd.DataFrame(
            {"a": [1.0, np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0, 4.0]},
            index=pd.date_range("2020-01-01", periods=4),
        )
        cleaned = clean_data(df, drop_threshold=0.5)
        assert "a" not in cleaned.columns
        assert "b" in cleaned.columns

    def test_invalid_method(self) -> None:
        """Test that invalid method raises ValueError."""
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Unknown method"):
            clean_data(df, method="invalid")


class TestResampleData:
    """Tests for resample_data function."""

    def test_weekly_resample(self, sample_prices: pd.DataFrame) -> None:
        """Test weekly resampling."""
        resampled = resample_data(sample_prices, freq="W")
        assert len(resampled) < len(sample_prices)

    def test_monthly_resample(self, sample_prices: pd.DataFrame) -> None:
        """Test monthly resampling."""
        resampled = resample_data(sample_prices, freq="ME")
        assert len(resampled) <= 12


class TestBaostockProvider:
    """Tests for BaostockProvider class structure."""

    def test_provider_instantiation(self) -> None:
        """Test that provider can be instantiated."""
        provider = BaostockProvider()
        assert provider._logged_in is False

    def test_provider_has_required_methods(self) -> None:
        """Test that provider has all required methods."""
        provider = BaostockProvider()
        assert hasattr(provider, "login")
        assert hasattr(provider, "logout")
        assert hasattr(provider, "get_k_data")
        assert hasattr(provider, "get_stock_list")
        assert hasattr(provider, "get_industry_classification")
