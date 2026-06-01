"""Factor-based trading strategies."""

from quant_portfolio.strategies.factor.value_factor import ValueFactor
from quant_portfolio.strategies.factor.size_factor import SizeFactor
from quant_portfolio.strategies.factor.quality_factor import QualityFactor
from quant_portfolio.strategies.factor.low_volatility import LowVolatility
from quant_portfolio.strategies.factor.dividend_yield import DividendYield
from quant_portfolio.strategies.factor.growth_factor import GrowthFactor
from quant_portfolio.strategies.factor.liquidity_factor import LiquidityFactor
from quant_portfolio.strategies.factor.beta_factor import BetaFactor
from quant_portfolio.strategies.factor.accruals_factor import AccrualsFactor
from quant_portfolio.strategies.factor.investment_factor import InvestmentFactor
from quant_portfolio.strategies.factor.multi_factor import MultiFactor
from quant_portfolio.strategies.factor.fundamental_factor import FundamentalFactor
from quant_portfolio.strategies.factor.technical_factor import TechnicalFactor
from quant_portfolio.strategies.factor.composite_alpha import CompositeAlpha
from quant_portfolio.strategies.factor.smart_beta import SmartBeta

__all__ = [
    "ValueFactor",
    "SizeFactor",
    "QualityFactor",
    "LowVolatility",
    "DividendYield",
    "GrowthFactor",
    "LiquidityFactor",
    "BetaFactor",
    "AccrualsFactor",
    "InvestmentFactor",
    "MultiFactor",
    "FundamentalFactor",
    "TechnicalFactor",
    "CompositeAlpha",
    "SmartBeta",
]
