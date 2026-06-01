"""Volatility-based trading strategies."""

from quant_portfolio.strategies.volatility.garch_vol import GARCHVol
from quant_portfolio.strategies.volatility.vol_targeting import VolTargeting
from quant_portfolio.strategies.volatility.vol_breakout import VolBreakout
from quant_portfolio.strategies.volatility.implied_realized_spread import ImpliedRealizedSpread
from quant_portfolio.strategies.volatility.vol_risk_premium import VolRiskPremium
from quant_portfolio.strategies.volatility.vol_regime import VolRegime
from quant_portfolio.strategies.volatility.vol_mean_reversion import VolMeanReversion
from quant_portfolio.strategies.volatility.vol_momentum import VolMomentum
from quant_portfolio.strategies.volatility.vol_surface import VolSurface
from quant_portfolio.strategies.volatility.straddle_strategy import StraddleStrategy

__all__ = [
    "GARCHVol",
    "VolTargeting",
    "VolBreakout",
    "ImpliedRealizedSpread",
    "VolRiskPremium",
    "VolRegime",
    "VolMeanReversion",
    "VolMomentum",
    "VolSurface",
    "StraddleStrategy",
]
