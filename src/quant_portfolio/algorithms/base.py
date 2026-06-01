"""Abstract Algorithm base class and AlgorithmRegistry for auto-discovery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import numpy as np
import pandas as pd


class Algorithm(ABC):
    """Abstract base class for all portfolio optimization algorithms.

    Subclasses must implement the `optimize` method which takes asset returns
    and returns an array of portfolio weights.
    """

    name: str = "BaseAlgorithm"

    def __init__(self, **params: Any) -> None:
        """Initialize algorithm with parameters.

        Parameters
        ----------
        **params : Any
            Algorithm-specific parameters.
        """
        self.params = params

    @abstractmethod
    def optimize(self, returns: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Optimize portfolio weights given asset returns.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of asset returns with DatetimeIndex.
        **kwargs : Any
            Additional optimization parameters.

        Returns
        -------
        np.ndarray
            Array of portfolio weights summing to approximately 1.0.
        """
        ...

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically register subclasses in the AlgorithmRegistry."""
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            AlgorithmRegistry.register(cls)


class AlgorithmRegistry:
    """Registry for auto-discovery of Algorithm implementations."""

    _registry: Dict[str, Type[Algorithm]] = {}

    @classmethod
    def register(cls, algorithm_class: Type[Algorithm]) -> None:
        """Register an algorithm class.

        Parameters
        ----------
        algorithm_class : Type[Algorithm]
            The algorithm class to register.
        """
        cls._registry[algorithm_class.name] = algorithm_class

    @classmethod
    def get(cls, name: str) -> Type[Algorithm]:
        """Get a registered algorithm by name.

        Parameters
        ----------
        name : str
            Name of the algorithm.

        Returns
        -------
        Type[Algorithm]
            The algorithm class.

        Raises
        ------
        KeyError
            If the algorithm is not registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Algorithm '{name}' not found. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_algorithms(cls) -> list[str]:
        """List all registered algorithm names.

        Returns
        -------
        list[str]
            List of registered algorithm names.
        """
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered algorithms."""
        cls._registry.clear()
