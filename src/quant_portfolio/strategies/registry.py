"""Strategy registry - auto-discovery and access to all strategies."""

from quant_portfolio.strategies.base import StrategyRegistry

# Import all strategy categories to trigger registration
import quant_portfolio.strategies.momentum  # noqa: F401
import quant_portfolio.strategies.mean_reversion  # noqa: F401
import quant_portfolio.strategies.trend_following  # noqa: F401
import quant_portfolio.strategies.stat_arb  # noqa: F401
import quant_portfolio.strategies.factor  # noqa: F401
import quant_portfolio.strategies.ml  # noqa: F401
import quant_portfolio.strategies.volatility  # noqa: F401
import quant_portfolio.strategies.technical  # noqa: F401


def list_strategies() -> list[str]:
    """List all registered strategy names.

    Returns
    -------
    list[str]
        List of all registered strategy names.
    """
    return StrategyRegistry.list_strategies()


def get_strategy(name: str):
    """Get a strategy class by name.

    Parameters
    ----------
    name : str
        Name of the strategy.

    Returns
    -------
    Type[Strategy]
        The strategy class.
    """
    return StrategyRegistry.get(name)
