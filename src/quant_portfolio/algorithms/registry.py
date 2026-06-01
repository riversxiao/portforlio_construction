"""Algorithm Registry - auto-discovery and access to all algorithms."""

# Import all algorithm modules to trigger registration
from quant_portfolio.algorithms.base import AlgorithmRegistry

# Import all categories to register algorithms
import quant_portfolio.algorithms.optimization  # noqa: F401
import quant_portfolio.algorithms.risk  # noqa: F401
import quant_portfolio.algorithms.allocation  # noqa: F401
import quant_portfolio.algorithms.signal_processing  # noqa: F401
import quant_portfolio.algorithms.statistical  # noqa: F401
import quant_portfolio.algorithms.numerical  # noqa: F401
import quant_portfolio.algorithms.ml_optimization  # noqa: F401


def list_algorithms() -> list[str]:
    """List all registered algorithm names.

    Returns
    -------
    list[str]
        List of all registered algorithm names.
    """
    return AlgorithmRegistry.list_algorithms()


def get_algorithm(name: str):
    """Get a registered algorithm class by name.

    Parameters
    ----------
    name : str
        Name of the algorithm.

    Returns
    -------
    Type[Algorithm]
        The algorithm class.
    """
    return AlgorithmRegistry.get(name)
