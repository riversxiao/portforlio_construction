"""Machine learning-based trading strategies."""

from quant_portfolio.strategies.ml.random_forest_strategy import RandomForestStrategy
from quant_portfolio.strategies.ml.gradient_boosting import GradientBoosting
from quant_portfolio.strategies.ml.lstm_strategy import LSTMStrategy, RecurrentWeightedStrategy
from quant_portfolio.strategies.ml.svm_strategy import SVMStrategy
from quant_portfolio.strategies.ml.kmeans_regime import KMeansRegime
from quant_portfolio.strategies.ml.pca_strategy import PCAStrategy
from quant_portfolio.strategies.ml.logistic_regression import LogisticRegressionStrategy
from quant_portfolio.strategies.ml.naive_bayes import NaiveBayesStrategy
from quant_portfolio.strategies.ml.decision_tree import DecisionTreeStrategy
from quant_portfolio.strategies.ml.ensemble_strategy import EnsembleStrategy

__all__ = [
    "RandomForestStrategy",
    "GradientBoosting",
    "LSTMStrategy",
    "RecurrentWeightedStrategy",
    "SVMStrategy",
    "KMeansRegime",
    "PCAStrategy",
    "LogisticRegressionStrategy",
    "NaiveBayesStrategy",
    "DecisionTreeStrategy",
    "EnsembleStrategy",
]
