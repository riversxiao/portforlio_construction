"""Abstract Strategy base class and StrategyRegistry for auto-discovery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import pandas as pd


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement the `generate_signals` method which takes
    market data and returns a DataFrame of trading signals.
    """

    name: str = "BaseStrategy"

    def __init__(self, **params: Any) -> None:
        """Initialize strategy with parameters.

        Parameters
        ----------
        **params : Any
            Strategy-specific parameters.
        """
        self.params = params

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from market data.

        Parameters
        ----------
        data : pd.DataFrame
            Market data (OHLCV) with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            DataFrame with at least a 'signal' column.
            Signal values: 1 (buy), -1 (sell), 0 (hold).
        """
        ...

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically register subclasses in the StrategyRegistry."""
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            StrategyRegistry.register(cls)


class StrategyRegistry:
    """Registry for auto-discovery of Strategy implementations."""

    _registry: Dict[str, Type[Strategy]] = {}

    @classmethod
    def register(cls, strategy_class: Type[Strategy]) -> None:
        """Register a strategy class.

        Parameters
        ----------
        strategy_class : Type[Strategy]
            The strategy class to register.
        """
        cls._registry[strategy_class.name] = strategy_class

    @classmethod
    def get(cls, name: str) -> Type[Strategy]:
        """Get a registered strategy by name.

        Parameters
        ----------
        name : str
            Name of the strategy.

        Returns
        -------
        Type[Strategy]
            The strategy class.

        Raises
        ------
        KeyError
            If the strategy is not registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Strategy '{name}' not found. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all registered strategy names.

        Returns
        -------
        list[str]
            List of registered strategy names.
        """
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered strategies."""
        cls._registry.clear()
