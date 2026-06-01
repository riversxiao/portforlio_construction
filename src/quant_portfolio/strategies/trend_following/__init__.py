"""Trend following trading strategies."""

from quant_portfolio.strategies.trend_following.moving_average_crossover import MovingAverageCrossover
from quant_portfolio.strategies.trend_following.turtle_trading import TurtleTrading
from quant_portfolio.strategies.trend_following.adx_trend import ADXTrend
from quant_portfolio.strategies.trend_following.supertrend import Supertrend
from quant_portfolio.strategies.trend_following.ichimoku import Ichimoku
from quant_portfolio.strategies.trend_following.parabolic_sar import ParabolicSAR
from quant_portfolio.strategies.trend_following.linear_regression_channel import LinearRegressionChannel
from quant_portfolio.strategies.trend_following.keltner_channel import KeltnerChannel
from quant_portfolio.strategies.trend_following.heikin_ashi import HeikinAshi
from quant_portfolio.strategies.trend_following.aroon_trend import AroonTrend

__all__ = [
    "MovingAverageCrossover",
    "TurtleTrading",
    "ADXTrend",
    "Supertrend",
    "Ichimoku",
    "ParabolicSAR",
    "LinearRegressionChannel",
    "KeltnerChannel",
    "HeikinAshi",
    "AroonTrend",
]
