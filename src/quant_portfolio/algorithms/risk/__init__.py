"""Risk management algorithms."""

from quant_portfolio.algorithms.risk.var_historical import VaRHistoricalAlgorithm
from quant_portfolio.algorithms.risk.var_parametric import VaRParametricAlgorithm
from quant_portfolio.algorithms.risk.var_monte_carlo import VaRMonteCarloAlgorithm
from quant_portfolio.algorithms.risk.cvar_calculation import CVaRCalculationAlgorithm
from quant_portfolio.algorithms.risk.stress_testing import StressTestingAlgorithm
from quant_portfolio.algorithms.risk.drawdown_control import DrawdownControlAlgorithm
from quant_portfolio.algorithms.risk.position_sizing import PositionSizingAlgorithm
from quant_portfolio.algorithms.risk.stop_loss import StopLossAlgorithm
from quant_portfolio.algorithms.risk.correlation_regime import CorrelationRegimeAlgorithm
from quant_portfolio.algorithms.risk.tail_risk import TailRiskAlgorithm
from quant_portfolio.algorithms.risk.risk_budgeting import RiskBudgetingAlgorithm
from quant_portfolio.algorithms.risk.marginal_risk import MarginalRiskAlgorithm
from quant_portfolio.algorithms.risk.factor_risk_decomposition import FactorRiskDecompositionAlgorithm
from quant_portfolio.algorithms.risk.liquidity_risk import LiquidityRiskAlgorithm
from quant_portfolio.algorithms.risk.concentration_risk import ConcentrationRiskAlgorithm

__all__ = [
    "VaRHistoricalAlgorithm",
    "VaRParametricAlgorithm",
    "VaRMonteCarloAlgorithm",
    "CVaRCalculationAlgorithm",
    "StressTestingAlgorithm",
    "DrawdownControlAlgorithm",
    "PositionSizingAlgorithm",
    "StopLossAlgorithm",
    "CorrelationRegimeAlgorithm",
    "TailRiskAlgorithm",
    "RiskBudgetingAlgorithm",
    "MarginalRiskAlgorithm",
    "FactorRiskDecompositionAlgorithm",
    "LiquidityRiskAlgorithm",
    "ConcentrationRiskAlgorithm",
]
