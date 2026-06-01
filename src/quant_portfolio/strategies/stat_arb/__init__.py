"""Statistical arbitrage trading strategies."""

from quant_portfolio.strategies.stat_arb.pca_stat_arb import PCAStatArb
from quant_portfolio.strategies.stat_arb.mean_field_arb import MeanFieldArb
from quant_portfolio.strategies.stat_arb.copula_arb import CopulaArb
from quant_portfolio.strategies.stat_arb.relative_value import RelativeValue
from quant_portfolio.strategies.stat_arb.etf_arb import ETFArb
from quant_portfolio.strategies.stat_arb.lead_lag import LeadLag
from quant_portfolio.strategies.stat_arb.dispersion_trading import DispersionTrading
from quant_portfolio.strategies.stat_arb.factor_neutral_arb import FactorNeutralArb
from quant_portfolio.strategies.stat_arb.regime_switching_arb import RegimeSwitchingArb
from quant_portfolio.strategies.stat_arb.microstructure_arb import MicrostructureArb

__all__ = [
    "PCAStatArb",
    "MeanFieldArb",
    "CopulaArb",
    "RelativeValue",
    "ETFArb",
    "LeadLag",
    "DispersionTrading",
    "FactorNeutralArb",
    "RegimeSwitchingArb",
    "MicrostructureArb",
]
