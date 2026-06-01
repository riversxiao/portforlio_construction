"""Mean reversion trading strategies."""

from quant_portfolio.strategies.mean_reversion.bollinger_bands import BollingerBands
from quant_portfolio.strategies.mean_reversion.pairs_trading import PairsTrading
from quant_portfolio.strategies.mean_reversion.ornstein_uhlenbeck import OrnsteinUhlenbeck
from quant_portfolio.strategies.mean_reversion.zscore_reversion import ZScoreReversion
from quant_portfolio.strategies.mean_reversion.kalman_filter_mr import KalmanFilterMR
from quant_portfolio.strategies.mean_reversion.johansen_cointegration import JohansenCointegration
from quant_portfolio.strategies.mean_reversion.half_life_mr import HalfLifeMR
from quant_portfolio.strategies.mean_reversion.rsi_reversion import RSIReversion
from quant_portfolio.strategies.mean_reversion.moving_average_reversion import MovingAverageReversion
from quant_portfolio.strategies.mean_reversion.mean_reversion_portfolio import MeanReversionPortfolio
from quant_portfolio.strategies.mean_reversion.adf_based import ADFBased
from quant_portfolio.strategies.mean_reversion.hurst_exponent import HurstExponent
from quant_portfolio.strategies.mean_reversion.variance_ratio import VarianceRatio
from quant_portfolio.strategies.mean_reversion.entropy_reversion import EntropyReversion
from quant_portfolio.strategies.mean_reversion.cointegration_basket import CointegrationBasket

__all__ = [
    "BollingerBands",
    "PairsTrading",
    "OrnsteinUhlenbeck",
    "ZScoreReversion",
    "KalmanFilterMR",
    "JohansenCointegration",
    "HalfLifeMR",
    "RSIReversion",
    "MovingAverageReversion",
    "MeanReversionPortfolio",
    "ADFBased",
    "HurstExponent",
    "VarianceRatio",
    "EntropyReversion",
    "CointegrationBasket",
]
