"""Momentum-based trading strategies."""

from quant_portfolio.strategies.momentum.simple_momentum import SimpleMomentum
from quant_portfolio.strategies.momentum.dual_momentum import DualMomentum
from quant_portfolio.strategies.momentum.momentum_acceleration import MomentumAcceleration
from quant_portfolio.strategies.momentum.sector_momentum import SectorMomentum
from quant_portfolio.strategies.momentum.earnings_momentum import EarningsMomentum
from quant_portfolio.strategies.momentum.rsi_momentum import RSIMomentum
from quant_portfolio.strategies.momentum.macd_momentum import MACDMomentum
from quant_portfolio.strategies.momentum.relative_strength import RelativeStrength
from quant_portfolio.strategies.momentum.time_series_momentum import TimeSeriesMomentum
from quant_portfolio.strategies.momentum.cross_sectional_momentum import CrossSectionalMomentum
from quant_portfolio.strategies.momentum.idiosyncratic_momentum import IdiosyncraticMomentum
from quant_portfolio.strategies.momentum.momentum_reversal import MomentumReversal
from quant_portfolio.strategies.momentum.volume_weighted_momentum import VolumeWeightedMomentum
from quant_portfolio.strategies.momentum.risk_adjusted_momentum import RiskAdjustedMomentum
from quant_portfolio.strategies.momentum.adaptive_momentum import AdaptiveMomentum

__all__ = [
    "SimpleMomentum",
    "DualMomentum",
    "MomentumAcceleration",
    "SectorMomentum",
    "EarningsMomentum",
    "RSIMomentum",
    "MACDMomentum",
    "RelativeStrength",
    "TimeSeriesMomentum",
    "CrossSectionalMomentum",
    "IdiosyncraticMomentum",
    "MomentumReversal",
    "VolumeWeightedMomentum",
    "RiskAdjustedMomentum",
    "AdaptiveMomentum",
]
