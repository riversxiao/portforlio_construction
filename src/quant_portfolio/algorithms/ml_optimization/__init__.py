"""Machine learning optimization algorithms."""

from quant_portfolio.algorithms.ml_optimization.neural_network_optimizer import NeuralNetworkOptimizerAlgorithm
from quant_portfolio.algorithms.ml_optimization.reinforcement_learning import ReinforcementLearningAlgorithm
from quant_portfolio.algorithms.ml_optimization.bayesian_optimization import BayesianOptimizationAlgorithm
from quant_portfolio.algorithms.ml_optimization.online_learning import OnlineLearningAlgorithm
from quant_portfolio.algorithms.ml_optimization.transfer_learning import TransferLearningAlgorithm
from quant_portfolio.algorithms.ml_optimization.attention_weighting import AttentionWeightingAlgorithm
from quant_portfolio.algorithms.ml_optimization.autoencoder_features import AutoencoderFeaturesAlgorithm
from quant_portfolio.algorithms.ml_optimization.clustering_allocation import ClusteringAllocationAlgorithm
from quant_portfolio.algorithms.ml_optimization.manifold_learning import ManifoldLearningAlgorithm
from quant_portfolio.algorithms.ml_optimization.graph_based import GraphBasedAlgorithm
from quant_portfolio.algorithms.ml_optimization.meta_learning import MetaLearningAlgorithm
from quant_portfolio.algorithms.ml_optimization.multi_objective import MultiObjectiveAlgorithm
from quant_portfolio.algorithms.ml_optimization.adversarial_robustness import AdversarialRobustnessAlgorithm
from quant_portfolio.algorithms.ml_optimization.variational_inference import VariationalInferenceAlgorithm
from quant_portfolio.algorithms.ml_optimization.gaussian_process import GaussianProcessAlgorithm

__all__ = [
    "NeuralNetworkOptimizerAlgorithm",
    "ReinforcementLearningAlgorithm",
    "BayesianOptimizationAlgorithm",
    "OnlineLearningAlgorithm",
    "TransferLearningAlgorithm",
    "AttentionWeightingAlgorithm",
    "AutoencoderFeaturesAlgorithm",
    "ClusteringAllocationAlgorithm",
    "ManifoldLearningAlgorithm",
    "GraphBasedAlgorithm",
    "MetaLearningAlgorithm",
    "MultiObjectiveAlgorithm",
    "AdversarialRobustnessAlgorithm",
    "VariationalInferenceAlgorithm",
    "GaussianProcessAlgorithm",
]
