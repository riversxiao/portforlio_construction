"""Configuration for alpha validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ValidationConfig:
    """Configuration for alpha validation thresholds and weights.

    Attributes
    ----------
    min_sharpe : float
        Minimum acceptable Sharpe ratio.
    min_sortino : float
        Minimum acceptable Sortino ratio.
    min_annualized_return : float
        Minimum acceptable annualized return.
    max_drawdown_threshold : float
        Maximum acceptable drawdown (negative number).
    max_cvar_95 : float
        Maximum acceptable CVaR at 95% confidence.
    max_annual_turnover : float
        Maximum acceptable annual turnover rate.
    min_hit_rate : float
        Minimum acceptable hit rate (fraction of winning periods).
    min_profit_factor : float
        Minimum acceptable profit factor.
    min_overall_score : float
        Minimum overall score to pass validation.
    dimension_weights : Dict[str, float]
        Weights for each dimension in overall score calculation.
    """

    min_sharpe: float = 0.5
    min_sortino: float = 0.7
    min_annualized_return: float = 0.05
    max_drawdown_threshold: float = -0.30
    max_cvar_95: float = 0.05
    max_annual_turnover: float = 20.0
    min_hit_rate: float = 0.45
    min_profit_factor: float = 1.0
    min_overall_score: float = 50.0
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        "return_ability": 1.0,
        "risk_adjusted": 1.0,
        "transaction_cost": 1.0,
        "drawdown_risk": 1.0,
        "efficiency": 1.0,
        "independence": 1.0,
        "coverage": 1.0,
        "robustness": 1.0,
    })
