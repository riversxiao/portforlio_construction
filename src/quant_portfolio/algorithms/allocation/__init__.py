"""Asset allocation algorithms."""

from quant_portfolio.algorithms.allocation.strategic_allocation import StrategicAllocationAlgorithm
from quant_portfolio.algorithms.allocation.tactical_allocation import TacticalAllocationAlgorithm
from quant_portfolio.algorithms.allocation.dynamic_allocation import DynamicAllocationAlgorithm
from quant_portfolio.algorithms.allocation.constant_proportion import ConstantProportionAlgorithm
from quant_portfolio.algorithms.allocation.target_date import TargetDateAlgorithm
from quant_portfolio.algorithms.allocation.momentum_allocation import MomentumAllocationAlgorithm
from quant_portfolio.algorithms.allocation.risk_on_off import RiskOnOffAlgorithm
from quant_portfolio.algorithms.allocation.min_correlation import MinCorrelationAlgorithm
from quant_portfolio.algorithms.allocation.max_decorrelation import MaxDecorrelationAlgorithm
from quant_portfolio.algorithms.allocation.core_satellite import CoreSatelliteAlgorithm
from quant_portfolio.algorithms.allocation.liability_driven import LiabilityDrivenAlgorithm
from quant_portfolio.algorithms.allocation.goal_based import GoalBasedAlgorithm
from quant_portfolio.algorithms.allocation.regime_based_allocation import RegimeBasedAllocationAlgorithm
from quant_portfolio.algorithms.allocation.volatility_weighted import VolatilityWeightedAlgorithm
from quant_portfolio.algorithms.allocation.drawdown_based import DrawdownBasedAlgorithm

__all__ = [
    "StrategicAllocationAlgorithm",
    "TacticalAllocationAlgorithm",
    "DynamicAllocationAlgorithm",
    "ConstantProportionAlgorithm",
    "TargetDateAlgorithm",
    "MomentumAllocationAlgorithm",
    "RiskOnOffAlgorithm",
    "MinCorrelationAlgorithm",
    "MaxDecorrelationAlgorithm",
    "CoreSatelliteAlgorithm",
    "LiabilityDrivenAlgorithm",
    "GoalBasedAlgorithm",
    "RegimeBasedAllocationAlgorithm",
    "VolatilityWeightedAlgorithm",
    "DrawdownBasedAlgorithm",
]
